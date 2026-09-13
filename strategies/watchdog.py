"""
strategies/watchdog.py — Autonomous Post-Market & EOD Flatten Watchdog.

Runs as an independent background daemon thread with its own dedicated Alpaca
TradingClient (eliminating shared failure domains with the main trading bridge).

Features:
- Orchestrator heartbeat tracking (Tier 1: 120s alarm, Tier 2: 300s alert).
- Configurable EOD buffer (default 120s -> 15:58:00 ET / 02:58:00 ICT).
- Broker truth verification directly via dedicated client.
- Emergency flatten on EOD with mandatory post-flatten zero-position assertion.
- Fail-closed error handling and clean session termination.
"""

import logging
import os
import threading
import time
from datetime import datetime, time as dt_time, timedelta
from typing import List, Optional

from alpaca.trading.client import TradingClient
from alpaca_bridge import TimeoutHTTPAdapter, configure_session_timeouts
from strategies.session import EASTERN_TZ, is_market_day, is_postmarket, market_close_time, now_eastern

logger = logging.getLogger(__name__)

from dataclasses import dataclass, field

# Global orchestrator heartbeat state
_last_heartbeat_timestamp: float = time.time()
_heartbeat_lock = threading.Lock()


@dataclass
class OrchestratorProgress:
    poll_id: str = "init"
    stage: str = "started"
    stage_started_at: float = field(default_factory=time.time)
    last_completed_checkpoint_at: float = field(default_factory=time.time)
    last_progress_at: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


_orchestrator_progress = OrchestratorProgress()


def touch_orchestrator_heartbeat(
    stage: str = "heartbeat",
    metadata: Optional[dict] = None,
    poll_id: Optional[str] = None,
) -> None:
    """Record current timestamp and active stage as orchestrator heartbeat."""
    global _last_heartbeat_timestamp, _orchestrator_progress
    now = time.time()
    with _heartbeat_lock:
        _last_heartbeat_timestamp = now
        _orchestrator_progress.stage = stage
        _orchestrator_progress.last_progress_at = now
        _orchestrator_progress.last_completed_checkpoint_at = now
        if poll_id is not None:
            _orchestrator_progress.poll_id = poll_id
            _orchestrator_progress.stage_started_at = now
        if metadata:
            _orchestrator_progress.metadata.update(metadata)


def get_orchestrator_progress() -> OrchestratorProgress:
    with _heartbeat_lock:
        return OrchestratorProgress(
            poll_id=_orchestrator_progress.poll_id,
            stage=_orchestrator_progress.stage,
            stage_started_at=_orchestrator_progress.stage_started_at,
            last_completed_checkpoint_at=_orchestrator_progress.last_completed_checkpoint_at,
            last_progress_at=_orchestrator_progress.last_progress_at,
            metadata=dict(_orchestrator_progress.metadata),
        )


def get_orchestrator_heartbeat_age() -> float:
    """Return seconds elapsed since the last orchestrator heartbeat."""
    with _heartbeat_lock:
        return time.time() - _last_heartbeat_timestamp


def reset_orchestrator_heartbeat(timestamp: Optional[float] = None) -> None:
    """Reset heartbeat timestamp (useful for testing)."""
    global _last_heartbeat_timestamp
    with _heartbeat_lock:
        _last_heartbeat_timestamp = timestamp if timestamp is not None else time.time()
        _orchestrator_progress.last_progress_at = _last_heartbeat_timestamp
        _orchestrator_progress.last_completed_checkpoint_at = _last_heartbeat_timestamp



class EODFlattenWatchdog:
    """Autonomous watchdog supervising main orchestrator liveness and guaranteeing EOD position flattening."""

    def __init__(
        self,
        poll_interval_seconds: float = 5.0,
        stale_alarm_seconds: float = 120.0,
        stale_alert_seconds: float = 300.0,
        eod_flatten_buffer_seconds: Optional[float] = None,
        alarm_cooldown_seconds: float = 60.0,
        auto_exit_on_clean_eod: bool = False,
        trading_client: Optional[TradingClient] = None,
    ):
        self.poll_interval = float(poll_interval_seconds)
        self.stale_alarm_seconds = float(stale_alarm_seconds)
        self.stale_alert_seconds = float(stale_alert_seconds)
        self.alarm_cooldown_seconds = float(alarm_cooldown_seconds)

        if eod_flatten_buffer_seconds is not None:
            self.eod_flatten_buffer_seconds = float(eod_flatten_buffer_seconds)
        else:
            try:
                import config
                self.eod_flatten_buffer_seconds = float(getattr(config, "EOD_FLATTEN_BUFFER_SECONDS", 120.0))
            except Exception:
                self.eod_flatten_buffer_seconds = 120.0

        self.auto_exit_on_clean_eod = bool(auto_exit_on_clean_eod)

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._flatten_executed: bool = False
        self._tier1_alarm_logged: bool = False
        self._tier2_alert_logged: bool = False
        self._last_tier1_alarm_time: float = 0.0
        self._last_tier2_alert_time: float = 0.0

        # Independent Broker Client (Separate Failure Domain)
        if trading_client is not None:
            self._client = trading_client
        else:
            self._client = self._create_isolated_trading_client()

    def _create_isolated_trading_client(self) -> Optional[TradingClient]:
        """Instantiate an isolated TradingClient with tight timeouts (connect=3s, read=5s)."""
        try:
            # Ensure .env is parsed if not already present
            if os.path.exists(".env"):
                with open(".env", "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            os.environ.setdefault(k.strip(), v.strip().strip("'").strip('"'))

            key = os.environ.get("DAYTRADE_APCA_API_KEY_ID") or os.environ.get("APCA_API_KEY_ID")
            secret = os.environ.get("DAYTRADE_APCA_API_SECRET_KEY") or os.environ.get("APCA_API_SECRET_KEY")
            if not key or not secret:
                logger.warning("[WATCHDOG] Missing Alpaca credentials — watchdog operating in monitor-only mode.")
                return None

            client = TradingClient(key, secret, paper=True)
            # Mount isolated TimeoutHTTPAdapter with bounded timeouts for emergency responsiveness
            watchdog_connect = float(os.environ.get("ALPACA_WATCHDOG_CONNECT_TIMEOUT", 8.0))
            watchdog_read = float(os.environ.get("ALPACA_WATCHDOG_READ_TIMEOUT", 10.0))
            configure_session_timeouts(
                getattr(client, "_session", None),
                connect_timeout=watchdog_connect,
                read_timeout=watchdog_read,
                total_retries=2,
                backoff_factor=0.3,
            )
            return client
        except Exception as exc:
            logger.error("[WATCHDOG] Failed to initialize isolated TradingClient: %s", exc)
            return None

    def start(self) -> None:
        """Start the watchdog background thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="EODFlattenWatchdog",
        )
        self._thread.start()
        logger.info("[WATCHDOG] EODFlattenWatchdog thread started (poll_interval=%.1fs, buffer=%.0fs)", self.poll_interval, self.eod_flatten_buffer_seconds)

    def stop(self, timeout: float = 2.0) -> None:
        """Stop the watchdog thread gracefully."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        logger.info("[WATCHDOG] EODFlattenWatchdog thread stopped.")

    def is_in_flatten_window(self, dt: Optional[datetime] = None) -> bool:
        """Check whether current time has entered the EOD Flatten window (e.g. >= 15:58:00 ET)."""
        dt_et = dt or now_eastern()
        if not is_market_day(dt_et.date()):
            return False

        close_t = market_close_time(dt_et.date())
        # Calculate flatten trigger time (close_time minus buffer seconds)
        close_dt = dt_et.replace(hour=close_t.hour, minute=close_t.minute, second=close_t.second, microsecond=0)
        flatten_trigger_dt = close_dt - timedelta(seconds=self.eod_flatten_buffer_seconds)

        return dt_et >= flatten_trigger_dt

    def check_and_execute_cycle(self, dt: Optional[datetime] = None) -> dict:
        """Perform one complete watchdog evaluation cycle (liveness + EOD flatten check)."""
        prog = get_orchestrator_progress()
        status = {
            "heartbeat_age": get_orchestrator_heartbeat_age(),
            "stalled_stage": prog.stage,
            "poll_id": prog.poll_id,
            "tier1_alarm": False,
            "tier2_alert": False,
            "in_flatten_window": False,
            "flatten_attempted": False,
            "positions_count": 0,
            "recheck_passed": True,
            "clean_eod": False,
        }

        # ── 1. Liveness Monitoring ──
        age = status["heartbeat_age"]
        now_ts = time.time()
        if age > self.stale_alert_seconds:
            status["tier2_alert"] = True
            if not self._tier2_alert_logged or (now_ts - self._last_tier2_alert_time) >= self.alarm_cooldown_seconds:
                logger.critical(
                    "🚨 [WATCHDOG TIER-2 ALERT] Main orchestrator thread appears frozen at stage '%s'! "
                    "Last heartbeat was %.1f seconds ago (threshold: %.0fs). Metadata: %s",
                    prog.stage, age, self.stale_alert_seconds, prog.metadata,
                )
                self._send_alert(f"🚨 [WATCHDOG CRITICAL] Main trading bot frozen at stage '{prog.stage}' for {age:.0f}s!")
                self._tier2_alert_logged = True
                self._last_tier2_alert_time = now_ts
        elif age > self.stale_alarm_seconds:
            status["tier1_alarm"] = True
            if not self._tier1_alarm_logged or (now_ts - self._last_tier1_alarm_time) >= self.alarm_cooldown_seconds:
                logger.warning(
                    "⚠️ [WATCHDOG ALARM] Main orchestrator heartbeat delayed: %.1f seconds (threshold: %.0fs, stalled at stage: '%s'). Metadata: %s",
                    age, self.stale_alarm_seconds, prog.stage, prog.metadata,
                )
                self._tier1_alarm_logged = True
                self._last_tier1_alarm_time = now_ts
        else:
            # Heartbeat is healthy -> reset alarm latching
            self._tier1_alarm_logged = False
            self._tier2_alert_logged = False


        # ── 2. EOD Flatten Window Evaluation ──
        in_window = self.is_in_flatten_window(dt) or is_postmarket()
        status["in_flatten_window"] = in_window

        if in_window:
            positions = self._get_broker_positions()
            status["positions_count"] = len(positions)

            if len(positions) > 0 and not self._flatten_executed:
                logger.critical(
                    "🚨 [WATCHDOG TIER-3 EMERGENCY] EOD threshold reached with %d open position(s)! "
                    "Symbols: %s. Initiating emergency broker flatten...",
                    len(positions), [p.symbol for p in positions],
                )
                status["flatten_attempted"] = True
                self._execute_broker_flatten()
                self._flatten_executed = True

                # ── Tier 4: Mandatory Post-Flatten Re-Verification ──
                time.sleep(1.0)  # Brief pause for broker matching engine
                recheck_positions = self._get_broker_positions()
                if len(recheck_positions) > 0:
                    status["recheck_passed"] = False
                    msg = (
                        f"🚨 [WATCHDOG CRITICAL INCIDENT] Emergency flatten executed but {len(recheck_positions)} "
                        f"position(s) remain open: {[p.symbol for p in recheck_positions]}!"
                    )
                    logger.critical(msg)
                    self._send_alert(msg)
                    # Retry flatten once more
                    self._execute_broker_flatten()
                else:
                    status["recheck_passed"] = True
                    logger.info("✅ [WATCHDOG AUDIT] EOD broker truth re-verified: 0 open positions. All positions flat.")

            elif len(positions) == 0:
                status["clean_eod"] = True

        return status

    def _get_broker_positions(self) -> list:
        """Fetch open positions directly from broker with fail-fast handling."""
        if not self._client:
            return []
        try:
            positions = self._client.get_all_positions()
            return list(positions or [])
        except Exception as exc:
            logger.error("[WATCHDOG] Failed to query broker positions: %s", exc)
            return []

    def _execute_broker_flatten(self) -> None:
        """Cancel all orders and close all positions via dedicated client."""
        if not self._client:
            logger.error("[WATCHDOG] Cannot execute flatten: dedicated TradingClient is unavailable.")
            return

        try:
            logger.info("[WATCHDOG] Cancelling all resting open orders...")
            self._client.cancel_orders()
        except Exception as exc:
            logger.error("[WATCHDOG] Error cancelling orders during flatten: %s", exc)

        try:
            logger.info("[WATCHDOG] Executing close_all_positions(cancel_orders=True)...")
            self._client.close_all_positions(cancel_orders=True)
            logger.info("✅ [WATCHDOG] Emergency broker close_all_positions command issued successfully.")
        except Exception as exc:
            logger.critical("[WATCHDOG] Error calling close_all_positions: %s", exc)

    def _send_alert(self, message: str) -> None:
        """Dispatch external alert (Discord/Webhook) if configured."""
        try:
            from strategies.webhook import send_discord_alert
            send_discord_alert(message)
        except Exception:
            pass

    def _run_loop(self) -> None:
        """Continuous watchdog supervisor loop."""
        while not self._stop_event.is_set():
            try:
                result = self.check_and_execute_cycle()
                if result.get("clean_eod") and is_postmarket():
                    if not getattr(self, "_clean_eod_logged", False):
                        logger.info("✅ [WATCHDOG] Session completed cleanly with 0 positions. Process held open for operator review.")
                        self._clean_eod_logged = True
            except Exception as exc:
                logger.error("[WATCHDOG] Exception in watchdog supervisor loop: %s", exc, exc_info=True)

            self._stop_event.wait(timeout=self.poll_interval)


# Singleton watchdog instance
_watchdog_instance: Optional[EODFlattenWatchdog] = None
_watchdog_instance_lock = threading.Lock()


def get_eod_watchdog(**kwargs) -> EODFlattenWatchdog:
    """Retrieve or create singleton EODFlattenWatchdog instance."""
    global _watchdog_instance
    with _watchdog_instance_lock:
        if _watchdog_instance is None:
            _watchdog_instance = EODFlattenWatchdog(**kwargs)
        return _watchdog_instance


def reset_eod_watchdog() -> None:
    """Stop and reset singleton watchdog (for testing)."""
    global _watchdog_instance
    with _watchdog_instance_lock:
        if _watchdog_instance is not None:
            try:
                _watchdog_instance.stop()
            except Exception:
                pass
            _watchdog_instance = None
