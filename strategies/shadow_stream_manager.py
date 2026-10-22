"""
strategies/shadow_stream_manager.py -- Alpaca IEX StockDataStream Shadow & Diagnostic Manager.

Phase 1 & 2 Implementation:
- Subscribes up to 30 symbols via Alpaca WebSocket StockDataStream (Basic Plan constraint).
- Retains 31st symbol strictly on the REST path with explicit audit logging.
- Subscribes to quotes, minute bars, and updatedBars (late trade corrections).
- Operates strictly in SHADOW / DIAGNOSTIC mode without mutating existing production gates.
- Enforces Fail-Closed health monitoring: flags unhealthy if stream disconnects or staleness threshold is exceeded.
- Maintains in-memory incremental bar cache and records comparison telemetry against REST.
"""

import os
import json
import time
import math
import logging
import asyncio
import threading
import random
import websockets
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

try:
    from alpaca.data.live.stock import StockDataStream
    from alpaca.data.enums import DataFeed
    ALPACA_STREAM_AVAILABLE = True
except ImportError:
    StockDataStream = None
    DataFeed = None
    ALPACA_STREAM_AVAILABLE = False

import config

logger = logging.getLogger(__name__)

MAX_WEBSOCKET_SYMBOLS = 30
MAX_CACHED_BARS_PER_SYMBOL = 1440  # 1440 1-min bars = 24h rolling buffer (covers full 04:00-20:00 ET extended hours)
DEFAULT_COMPARISON_LOG = Path(getattr(config, "LOG_DIR", "logs")) / "shadow_stream_comparison.jsonl"


@dataclass
class SymbolStreamMetadata:
    symbol: str
    source: str  # "STREAM" (top 30) or "REST_FALLBACK" (31st+)
    stream_status: str = "INITIALIZING"  # "ACTIVE", "STALE", "DISCONNECTED", "FALLBACK"
    last_event_timestamp: Optional[float] = None
    last_bar_timestamp: Optional[float] = None
    quotes_count: int = 0
    bars_count: int = 0


@dataclass
class StreamBarRecord:
    symbol: str
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float
    trade_count: int = 0
    vwap: Optional[float] = None
    is_updated: bool = False
    session_phase: str = "REGULAR"  # "PREMARKET" (04:00-09:30 ET), "REGULAR", "POSTMARKET"
    received_at: float = field(default_factory=time.time)
    event_timestamp: Optional[float] = None
    receive_delay_ms: Optional[float] = None
    handler_duration_ms: Optional[float] = None


@dataclass
class StreamQuoteRecord:
    symbol: str
    bid_price: float
    bid_size: float
    ask_price: float
    ask_size: float
    timestamp: float
    spread: float
    spread_pct: float
    received_at: float = field(default_factory=time.time)
    event_timestamp: Optional[float] = None
    receive_delay_ms: Optional[float] = None
    handler_duration_ms: Optional[float] = None


@dataclass
class StreamHealthStatus:
    is_connected: bool = False
    last_heartbeat: float = 0.0
    reconnect_count: int = 0
    quotes_received: int = 0
    bars_received: int = 0
    updated_bars_received: int = 0
    errors_count: int = 0
    streamed_symbols_count: int = 0
    rest_fallback_symbols_count: int = 0
    last_error: Optional[str] = None



if ALPACA_STREAM_AVAILABLE and StockDataStream is not None:
    class _HardenedStockDataStream(StockDataStream):
        """
        Phase B Tier 2 Hardened StockDataStream subclass.
        
        Upstream SDK DataStream._run_forever() has an unthrottled 0ms retry loop on
        unexpected exceptions (such as socket.gaierror on DNS failures during Wi-Fi drops),
        which floods logs with tens of thousands of tracebacks.
        
        This hardened stream enforces:
        1. Canonical Phase B Reconnect Backoff:
           2s -> 4s -> 8s -> 16s -> 30s (+-20% jitter)
           Preserving the mandatory 16s step without skipping.
        2. Fail-Closed Health Update:
           Transitions manager.health.is_connected to False immediately upon disconnect.
        3. Idempotent cleanup:
           Calls await self.close() on all error branches.
        4. Interruptible sleep:
           Checks self._should_run in small 0.2s slices during backoff, allowing stop()
           to exit cleanly without blocking.
        """
        def __init__(self, *args, manager: Optional[Any] = None, **kwargs):
            super().__init__(*args, **kwargs)
            self._manager = manager
            self._reconnect_attempt = 0
            self._base_delay = 2.0
            self._max_delay = 30.0
            self._jitter_ratio = 0.20

        async def _run_forever(self) -> None:
            self._loop = asyncio.get_running_loop()
            while not any(
                v
                for k, v in self._handlers.items()
                if k not in ("cancelErrors", "corrections")
            ):
                if not self._stop_stream_queue.empty():
                    self._stop_stream_queue.get(timeout=1)
                    return
                await asyncio.sleep(0.05)

            logger.info("[SHADOW STREAM] Started shadow market data stream event loop")
            self._should_run = True
            self._running = False

            while self._should_run:
                try:
                    if not self._should_run:
                        logger.info("[SHADOW STREAM] Shadow stream stopped gracefully (intentional)")
                        return

                    if not self._running:
                        logger.info("[SHADOW STREAM] Initiating data websocket connection (attempt %d)", self._reconnect_attempt + 1)
                        await self._start_ws()
                        await self._send_subscribe_msg()
                        self._running = True
                        self._reconnect_attempt = 0
                        if self._manager is not None:
                            with self._manager._lock:
                                self._manager.health.is_connected = True

                    await self._consume()

                except websockets.WebSocketException as wse:
                    await self.close()
                    self._running = False
                    self._mark_manager_disconnected(str(wse))
                    if not self._should_run:
                        break
                    await self._handle_backoff(type(wse).__name__, str(wse))

                except ValueError as ve:
                    if "insufficient subscription" in str(ve):
                        await self.close()
                        self._running = False
                        self._mark_manager_disconnected(str(ve))
                        logger.error("[SHADOW STREAM] Insufficient subscription: %s", ve)
                        return
                    await self.close()
                    self._running = False
                    self._mark_manager_disconnected(str(ve))
                    if not self._should_run:
                        break
                    await self._handle_backoff(type(ve).__name__, str(ve))

                except Exception as exc:
                    await self.close()
                    self._running = False
                    self._mark_manager_disconnected(str(exc))
                    if not self._should_run:
                        break
                    await self._handle_backoff(type(exc).__name__, str(exc))

                finally:
                    await asyncio.sleep(0)

        def _mark_manager_disconnected(self, err_msg: str) -> None:
            if self._manager is not None:
                with self._manager._lock:
                    self.health_is_conn = False
                    self._manager.health.is_connected = False
                    self._manager.health.last_error = err_msg
                    self._manager.health.errors_count += 1

        async def _handle_backoff(self, error_type: str, error_msg: str) -> None:
            self._reconnect_attempt += 1
            raw_delay = min(self._max_delay, self._base_delay * (2 ** (self._reconnect_attempt - 1)))
            delay = raw_delay * (1.0 + random.uniform(-self._jitter_ratio, self._jitter_ratio))
            delay = max(0.5, delay)

            logger.warning(
                "[SHADOW STREAM RECONNECT] %s: %s | Reconnect attempt %d backing off for %.2fs",
                error_type,
                error_msg,
                self._reconnect_attempt,
                delay,
            )

            end_time = time.monotonic() + delay
            while time.monotonic() < end_time:
                if not self._should_run:
                    break
                await asyncio.sleep(0.2)
else:
    class _HardenedStockDataStream:
        def __init__(self, *args, **kwargs):
            pass


class ShadowStreamManager:
    """
    Isolated Shadow Data Stream Manager for Alpaca IEX WebSocket data.
    Provides diagnostic observation and incremental bar caching without modifying live trading decisions.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        watchlist: Optional[List[str]] = None,
        max_stream_symbols: int = MAX_WEBSOCKET_SYMBOLS,
        comparison_log_path: Optional[Path] = None,
    ):
        self._lock = threading.RLock()
        self.api_key = api_key or os.environ.get("DAYTRADE_APCA_API_KEY_ID", "") or os.environ.get("APCA_API_KEY_ID", "")
        self.secret_key = secret_key or os.environ.get("DAYTRADE_APCA_API_SECRET_KEY", "") or os.environ.get("APCA_API_SECRET_KEY", "")
        
        all_symbols = [s.upper().strip() for s in (watchlist or getattr(config, "DAYTRADE_WATCHLIST", []))]
        
        # Partition: Top 30 for WebSocket, 31st+ for REST Fallback (Basic Plan Compliance)
        self.streamed_symbols = all_symbols[:max_stream_symbols]
        self.rest_fallback_symbols = all_symbols[max_stream_symbols:]
        
        self.comparison_log_path = comparison_log_path or DEFAULT_COMPARISON_LOG
        self.comparison_log_path.parent.mkdir(parents=True, exist_ok=True)

        self._quotes: Dict[str, StreamQuoteRecord] = {}
        self._bars: Dict[str, List[StreamBarRecord]] = {sym: [] for sym in self.streamed_symbols}
        self.symbol_metadata: Dict[str, SymbolStreamMetadata] = {}
        for sym in self.streamed_symbols:
            self.symbol_metadata[sym] = SymbolStreamMetadata(symbol=sym, source="STREAM", stream_status="INITIALIZING")
        for sym in self.rest_fallback_symbols:
            self.symbol_metadata[sym] = SymbolStreamMetadata(symbol=sym, source="REST_FALLBACK", stream_status="FALLBACK")

        self.health = StreamHealthStatus(
            streamed_symbols_count=len(self.streamed_symbols),
            rest_fallback_symbols_count=len(self.rest_fallback_symbols),
        )

        self._stream: Optional[Any] = None
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._running = False

        logger.info(
            "[SHADOW STREAM INITIALIZED] Streaming %d symbols on IEX WebSocket; %d symbols reserved on REST fallback: %s",
            len(self.streamed_symbols),
            len(self.rest_fallback_symbols),
            self.rest_fallback_symbols,
        )

    # -------------------------------------------------------------------------
    # WebSocket Event Handlers
    # -------------------------------------------------------------------------

    async def _handle_quote(self, quote: Any) -> None:
        """Handle incoming streaming quote event with precise latency measurement."""
        t0 = time.perf_counter()
        received_at = time.time()
        try:
            sym = str(getattr(quote, "symbol", "")).upper().strip()
            bid_p = float(getattr(quote, "bid_price", 0.0) or 0.0)
            bid_s = float(getattr(quote, "bid_size", 0.0) or 0.0)
            ask_p = float(getattr(quote, "ask_price", 0.0) or 0.0)
            ask_s = float(getattr(quote, "ask_size", 0.0) or 0.0)
            
            ts_raw = getattr(quote, "timestamp", None)
            if hasattr(ts_raw, "timestamp"):
                ts = ts_raw.timestamp()
            elif isinstance(ts_raw, (int, float)):
                ts = float(ts_raw)
            else:
                ts = received_at

            receive_delay_ms = max(0.0, (received_at - ts) * 1000.0)
            mid = (bid_p + ask_p) / 2.0 if (bid_p > 0 and ask_p > 0) else (ask_p or bid_p or 0.0)
            spread = max(0.0, ask_p - bid_p) if (bid_p > 0 and ask_p > 0) else 0.0
            spread_pct = (spread / mid * 100.0) if mid > 0 else 0.0
            handler_duration_ms = (time.perf_counter() - t0) * 1000.0

            record = StreamQuoteRecord(
                symbol=sym,
                bid_price=bid_p,
                bid_size=bid_s,
                ask_price=ask_p,
                ask_size=ask_s,
                timestamp=ts,
                spread=spread,
                spread_pct=spread_pct,
                received_at=received_at,
                event_timestamp=ts,
                receive_delay_ms=round(receive_delay_ms, 2),
                handler_duration_ms=round(handler_duration_ms, 3),
            )

            with self._lock:
                self._quotes[sym] = record
                meta = self.symbol_metadata.get(sym)
                if meta:
                    meta.last_event_timestamp = ts
                    meta.quotes_count += 1
                    meta.stream_status = "ACTIVE"
                self.health.quotes_received += 1
                self.health.last_heartbeat = time.time()
                self.health.is_connected = True
        except Exception as exc:
            with self._lock:
                self.health.errors_count += 1
                self.health.last_error = f"Quote handle error: {exc}"
            logger.debug("[SHADOW STREAM] Error processing quote: %s", exc)

    async def _handle_bar(self, bar: Any, is_update: bool = False) -> None:
        """Handle regular 1m bar and updatedBar (late trades correction) with deduplication."""
        t0 = time.perf_counter()
        received_at = time.time()
        try:
            sym = str(getattr(bar, "symbol", "")).upper().strip()
            o = float(getattr(bar, "open", 0.0) or 0.0)
            h = float(getattr(bar, "high", 0.0) or 0.0)
            l = float(getattr(bar, "low", 0.0) or 0.0)
            c = float(getattr(bar, "close", 0.0) or 0.0)
            v = float(getattr(bar, "volume", 0.0) or 0.0)
            cnt = int(getattr(bar, "trade_count", 0) or 0)
            vw = getattr(bar, "vwap", None)
            vwap = float(vw) if vw is not None else None

            ts_raw = getattr(bar, "timestamp", None)
            if hasattr(ts_raw, "timestamp"):
                ts = ts_raw.timestamp()
            elif isinstance(ts_raw, (int, float)):
                ts = float(ts_raw)
            else:
                ts = received_at

            receive_delay_ms = max(0.0, (received_at - ts) * 1000.0)
            dt_utc = datetime.fromtimestamp(ts, tz=timezone.utc)
            et_hour = (dt_utc.hour - 4) % 24
            et_min = dt_utc.minute
            et_time_dec = et_hour + et_min / 60.0
            if et_time_dec < 9.5:
                phase = "PREMARKET"
            elif et_time_dec <= 16.0:
                phase = "REGULAR"
            else:
                phase = "POSTMARKET"

            handler_duration_ms = (time.perf_counter() - t0) * 1000.0

            record = StreamBarRecord(
                symbol=sym,
                timestamp=ts,
                open=o,
                high=h,
                low=l,
                close=c,
                volume=v,
                trade_count=cnt,
                vwap=vwap,
                is_updated=is_update,
                session_phase=phase,
                received_at=received_at,
                event_timestamp=ts,
                receive_delay_ms=round(receive_delay_ms, 2),
                handler_duration_ms=round(handler_duration_ms, 3),
            )

            with self._lock:
                bar_list = self._bars.setdefault(sym, [])
                # Robust Deduplication & In-place Correction:
                # Find if a bar with the same minute timestamp window already exists
                existing_idx = None
                for idx in reversed(range(len(bar_list))):
                    if abs(bar_list[idx].timestamp - ts) < 30.0:  # Same 1-minute window
                        existing_idx = idx
                        break

                if existing_idx is not None:
                    bar_list[existing_idx] = record
                    if is_update:
                        self.health.updated_bars_received += 1
                else:
                    bar_list.append(record)
                    if is_update:
                        self.health.updated_bars_received += 1
                    else:
                        self.health.bars_received += 1

                # Rolling buffer: 1440 bars = 24h of 1m bars (preserves premarket 04:00-09:30 ET bars)
                if len(bar_list) > MAX_CACHED_BARS_PER_SYMBOL:
                    del bar_list[:len(bar_list) - MAX_CACHED_BARS_PER_SYMBOL]

                meta = self.symbol_metadata.get(sym)
                if meta:
                    meta.last_bar_timestamp = ts
                    meta.bars_count = len(bar_list)
                    meta.stream_status = "ACTIVE"

                self.health.last_heartbeat = time.time()
                self.health.is_connected = True
        except Exception as exc:
            with self._lock:
                self.health.errors_count += 1
                self.health.last_error = f"Bar handle error: {exc}"
            logger.debug("[SHADOW STREAM] Error processing bar: %s", exc)

    async def _handle_updated_bar(self, bar: Any) -> None:
        """Dedicated handler for updatedBars channel."""
        await self._handle_bar(bar, is_update=True)

    # -------------------------------------------------------------------------
    # Lifecycle Control
    # -------------------------------------------------------------------------

    def start(self) -> bool:
        """Start the background WebSocket stream thread in non-blocking fashion."""
        if not ALPACA_STREAM_AVAILABLE:
            logger.warning("[SHADOW STREAM BLOCKED] alpaca-py StockDataStream module not available.")
            return False

        if not self.api_key or not self.secret_key:
            self.api_key = (
                os.environ.get("DAYTRADE_APCA_API_KEY_ID", "")
                or os.environ.get("APCA_API_KEY_ID", "")
            ).strip()
            self.secret_key = (
                os.environ.get("DAYTRADE_APCA_API_SECRET_KEY", "")
                or os.environ.get("APCA_API_SECRET_KEY", "")
            ).strip()

        if not self.api_key or not self.secret_key:
            logger.warning("[SHADOW STREAM BLOCKED] Alpaca API credentials not provided.")
            return False

        if self._running:
            return True

        self._running = True
        self._thread = threading.Thread(target=self._run_stream_worker, name="ShadowStreamWorker", daemon=True)
        self._thread.start()
        logger.info("[SHADOW STREAM] Background stream worker initiated.")
        return True

    def bootstrap_historical_bars(self, symbols: Optional[List[str]] = None, lookback_hours: int = 16) -> Dict[str, int]:
        """
        Bootstrap the 1,440-bar ring buffer with historical 1m bars from Alpaca IEX feed.
        Ensures premarket (04:00-09:30 ET) data is present upon startup for GAP_AND_GO and indicators.
        """
        target_symbols = symbols or self.streamed_symbols
        loaded_counts: Dict[str, int] = {}
        if not target_symbols:
            return loaded_counts

        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
            from alpaca.data.enums import DataFeed

            api_k = self.api_key or os.environ.get("DAYTRADE_APCA_API_KEY_ID") or os.environ.get("APCA_API_KEY_ID")
            api_s = self.secret_key or os.environ.get("DAYTRADE_APCA_API_SECRET_KEY") or os.environ.get("APCA_API_SECRET_KEY")
            if not api_k or not api_s:
                logger.debug("[SHADOW STREAM BOOTSTRAP] Credentials missing, skipping bootstrap.")
                return loaded_counts

            client = StockHistoricalDataClient(api_k, api_s)
            start_dt = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)

            req = StockBarsRequest(
                symbol_or_symbols=target_symbols,
                timeframe=TimeFrame.Minute,
                start=start_dt,
                feed=DataFeed.IEX,
            )
            bars_resp = client.get_stock_bars(req)

            with self._lock:
                for sym in target_symbols:
                    if sym not in bars_resp:
                        loaded_counts[sym] = len(self._bars.get(sym, []))
                        continue
                    sym_bars = bars_resp[sym]
                    bar_list = self._bars.setdefault(sym, [])
                    existing_ts = {int(b.timestamp // 60) for b in bar_list}

                    for raw_b in sym_bars:
                        raw_ts = getattr(raw_b, "timestamp", None)
                        if hasattr(raw_ts, "timestamp"):
                            ts = raw_ts.timestamp()
                        elif isinstance(raw_ts, (int, float)):
                            ts = float(raw_ts)
                        else:
                            continue

                        min_key = int(ts // 60)
                        if min_key in existing_ts:
                            continue

                        dt_utc = datetime.fromtimestamp(ts, tz=timezone.utc)
                        et_hour = (dt_utc.hour - 4) % 24
                        et_min = dt_utc.minute
                        et_time_dec = et_hour + et_min / 60.0
                        phase = "PREMARKET" if et_time_dec < 9.5 else ("REGULAR" if et_time_dec <= 16.0 else "POSTMARKET")

                        rec = StreamBarRecord(
                            symbol=sym,
                            timestamp=ts,
                            open=float(getattr(raw_b, "open", 0.0) or 0.0),
                            high=float(getattr(raw_b, "high", 0.0) or 0.0),
                            low=float(getattr(raw_b, "low", 0.0) or 0.0),
                            close=float(getattr(raw_b, "close", 0.0) or 0.0),
                            volume=float(getattr(raw_b, "volume", 0.0) or 0.0),
                            trade_count=int(getattr(raw_b, "trade_count", 0) or 0),
                            vwap=float(raw_b.vwap) if getattr(raw_b, "vwap", None) is not None else None,
                            session_phase=phase,
                            received_at=time.time(),
                            event_timestamp=ts,
                            receive_delay_ms=0.0,
                            handler_duration_ms=0.0,
                        )
                        bar_list.append(rec)
                        existing_ts.add(min_key)

                    bar_list.sort(key=lambda b: b.timestamp)
                    if len(bar_list) > MAX_CACHED_BARS_PER_SYMBOL:
                        del bar_list[:len(bar_list) - MAX_CACHED_BARS_PER_SYMBOL]

                    loaded_counts[sym] = len(bar_list)
                    meta = self.symbol_metadata.get(sym)
                    if meta and bar_list:
                        meta.last_bar_timestamp = bar_list[-1].timestamp
                        meta.bars_count = len(bar_list)

            logger.info(
                "[SHADOW STREAM BOOTSTRAP] Bootstrapped historical 1m bars for %d symbols.",
                len(loaded_counts),
            )
        except Exception as exc:
            logger.debug("[SHADOW STREAM BOOTSTRAP] Historical bar bootstrap note: %s", exc)

        return loaded_counts

    def _run_stream_worker(self) -> None:
        """Internal worker running asyncio loop for StockDataStream."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        # Trigger bootstrap before starting stream ingestion
        try:
            self.bootstrap_historical_bars()
        except Exception as b_exc:
            logger.debug("[SHADOW STREAM] Bootstrap non-blocking exception: %s", b_exc)

        try:
            stream_cls = _HardenedStockDataStream if ALPACA_STREAM_AVAILABLE and StockDataStream is not None else StockDataStream
            self._stream = stream_cls(
                api_key=self.api_key,
                secret_key=self.secret_key,
                feed=DataFeed.IEX,
                manager=self,
            )

            # Subscribe to quotes, bars, and updatedBars for top 30 symbols
            if self.streamed_symbols:
                self._stream.subscribe_quotes(self._handle_quote, *self.streamed_symbols)
                self._stream.subscribe_bars(self._handle_bar, *self.streamed_symbols)
                self._stream.subscribe_updated_bars(self._handle_updated_bar, *self.streamed_symbols)

            logger.info("[SHADOW STREAM RUNNING] Subscriptions registered for %d symbols.", len(self.streamed_symbols))
            with self._lock:
                self.health.is_connected = True

            self._stream.run()
        except Exception as exc:
            with self._lock:
                self.health.is_connected = False
                self.health.last_error = str(exc)
                self.health.errors_count += 1
            logger.error("[SHADOW STREAM TERMINATED] Stream encountered fatal error: %s", exc)
        finally:
            with self._lock:
                self.health.is_connected = False
            self._running = False

    def stop(self) -> None:
        """Gracefully shut down the background stream."""
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception:
                pass
        with self._lock:
            self.health.is_connected = False
        logger.info("[SHADOW STREAM STOPPED] Stream gracefully shut down.")

    # -------------------------------------------------------------------------
    # Fail-Closed Health & Cache Queries
    # -------------------------------------------------------------------------

    def is_healthy(self, max_heartbeat_age_seconds: float = 60.0) -> bool:
        """
        Check stream health strictly fail-closed.
        Returns False if disconnected, heartbeat too old, or stream uninitialized.
        """
        with self._lock:
            if not self.health.is_connected:
                return False
            if self.health.last_heartbeat <= 0:
                return False
            age = time.time() - self.health.last_heartbeat
            return age <= max_heartbeat_age_seconds

    def get_latest_quote(self, symbol: str) -> Optional[StreamQuoteRecord]:
        """Fetch cached latest streaming quote for a symbol."""
        sym = symbol.upper().strip()
        with self._lock:
            return self._quotes.get(sym)

    def get_bars(self, symbol: str, limit: int = 100) -> List[StreamBarRecord]:
        """Fetch cached incremental 1m bars for a symbol."""
        sym = symbol.upper().strip()
        with self._lock:
            bars = self._bars.get(sym, [])
            return list(bars[-limit:])

    def get_health_status(self) -> Dict[str, Any]:
        """Get structured snapshot of stream health metrics."""
        with self._lock:
            hb_age = (time.time() - self.health.last_heartbeat) if self.health.last_heartbeat > 0 else None
            return {
                "is_connected": self.health.is_connected,
                "is_healthy": self.is_healthy(),
                "last_heartbeat_timestamp": self.health.last_heartbeat if self.health.last_heartbeat > 0 else None,
                "heartbeat_age_seconds": round(hb_age, 2) if hb_age is not None else None,
                "reconnect_count": self.health.reconnect_count,
                "quotes_count": self.health.quotes_received,
                "bars_count": self.health.bars_received,
                "updated_bars_count": self.health.updated_bars_received,
                "errors_count": self.health.errors_count,
                "last_error": self.health.last_error,
                "streamed_symbols_count": self.health.streamed_symbols_count,
                "rest_fallback_symbols_count": self.health.rest_fallback_symbols_count,
            }

    def get_summary(self) -> Dict[str, Any]:
        """Get structured summary of stream health and symbol metadata."""
        with self._lock:
            return {
                "health": self.get_health_status(),
                "symbol_metadata": {
                    sym: {
                        "symbol": meta.symbol,
                        "source": meta.source,
                        "stream_status": meta.stream_status,
                        "last_event_timestamp": meta.last_event_timestamp,
                        "last_bar_timestamp": meta.last_bar_timestamp,
                        "quotes_count": meta.quotes_count,
                        "bars_count": meta.bars_count,
                    }
                    for sym, meta in self.symbol_metadata.items()
                },
            }

    def record_shadow_comparison(
        self,
        symbol: str,
        rest_quote: Optional[dict],
        rest_latency_ms: float,
        is_spread_veto: bool = False,
        cycle_id: Optional[str] = None,
        rest_quote_timestamp: Optional[Any] = None,
    ) -> None:
        """
        Record side-by-side comparison between live REST execution quote and cached WebSocket stream quote.
        Persisted to logs/shadow_stream_comparison.jsonl for Phase 3 audit.
        """
        sym = symbol.upper().strip()
        stream_q = self.get_latest_quote(sym)
        now_ts = time.time()
        
        stream_bid = stream_q.bid_price if stream_q else None
        stream_ask = stream_q.ask_price if stream_q else None
        stream_spread = stream_q.spread if stream_q else None
        stream_spread_pct = stream_q.spread_pct if stream_q else None
        stream_event_ts = stream_q.event_timestamp if stream_q else None
        stream_received_at = stream_q.received_at if stream_q else None
        stream_receive_delay_ms = stream_q.receive_delay_ms if stream_q else None
        stream_handler_ms = stream_q.handler_duration_ms if stream_q else None
        stream_quote_age_ms = ((now_ts - stream_q.timestamp) * 1000.0) if stream_q else None

        rest_bid = rest_quote.get("bid") if rest_quote else None
        rest_ask = rest_quote.get("ask") if rest_quote else None
        rest_ts = rest_quote_timestamp or (rest_quote.get("timestamp") if rest_quote else None)
        rest_spread_pct = None
        if rest_bid and rest_ask and rest_bid > 0 and rest_ask > 0:
            mid = (rest_bid + rest_ask) / 2.0
            rest_spread_pct = ((rest_ask - rest_bid) / mid * 100.0) if mid > 0 else None

        rest_age_ms = None
        if rest_ts:
            if hasattr(rest_ts, "timestamp"):
                rest_age_ms = max(0.0, (now_ts - rest_ts.timestamp()) * 1000.0)
            elif isinstance(rest_ts, (int, float)):
                rest_age_ms = max(0.0, (now_ts - float(rest_ts)) * 1000.0)

        spread_diff_pct = None
        if rest_spread_pct is not None and stream_spread_pct is not None:
            spread_diff_pct = round(rest_spread_pct - stream_spread_pct, 4)

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cycle_id": cycle_id or f"eval_{int(now_ts * 1000)}",
            "symbol": sym,
            "is_streamed": sym in self.streamed_symbols,
            "is_spread_veto": is_spread_veto,
            "rest": {
                "source": "REST_IEX",
                "bid": rest_bid,
                "ask": rest_ask,
                "spread_pct": round(rest_spread_pct, 4) if rest_spread_pct is not None else None,
                "latency_ms": round(rest_latency_ms, 2),
                "quote_timestamp": rest_ts if isinstance(rest_ts, (int, float, str)) else str(rest_ts),
                "quote_age_ms": round(rest_age_ms, 2) if rest_age_ms is not None else None,
            },
            "stream": {
                "source": "WS_IEX",
                "bid": stream_bid,
                "ask": stream_ask,
                "spread_pct": round(stream_spread_pct, 4) if stream_spread_pct is not None else None,
                "event_timestamp": stream_event_ts,
                "received_at": stream_received_at,
                "receive_delay_ms": stream_receive_delay_ms,
                "handler_duration_ms": stream_handler_ms,
                "quote_age_ms": round(stream_quote_age_ms, 2) if stream_quote_age_ms is not None else None,
                "stream_healthy": self.is_healthy(),
                "reconnect_count": self.health.reconnect_count,
            },
            "delta": {
                "spread_diff_pct": spread_diff_pct,
            },
        }

        try:
            with open(self.comparison_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as exc:
            logger.debug("[SHADOW STREAM] Failed to write comparison log: %s", exc)


# Global singleton instance for non-intrusive shadow attachment
shadow_stream_manager: Optional[ShadowStreamManager] = None


def get_shadow_stream_manager() -> ShadowStreamManager:
    """Retrieve or lazily instantiate global ShadowStreamManager singleton."""
    global shadow_stream_manager
    if shadow_stream_manager is None:
        shadow_stream_manager = ShadowStreamManager()
    return shadow_stream_manager

