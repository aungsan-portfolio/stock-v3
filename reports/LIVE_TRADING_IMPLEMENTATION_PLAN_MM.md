# Live Trading သို့ ပြောင်းရန် — လုံခြုံစွာ အဆင့်ဆင့် အကောင်အထည်ဖော်ရေး Plan (မြန်မာ)

> တွဲဖက်စာတမ်း: `reports/LIVE_TRADING_READINESS_MM.md` (verified findings)
> ဤစာတမ်းက အဲဒီ finding တွေကို **code-level တာဝန်**, **paper-test gate**, **go/no-go criteria** အဖြစ်
> ပြောင်းပေးထားသည်။
>
> **ရွှေစည်းမျဉ်း:** Phase ၆ (live conversion) မရောက်မချင်း — `COACH_LIVE_TRADING_ENABLED=False`,
> `IBKR_PORT=7497`, `REQUIRE_PAPER_PORT=True` တို့ကို **လုံးဝ မပြောင်းရ**။ ဒီ plan တစ်ခုလုံးကို
> **paper account ပေါ်မှာပဲ** ဆောက်ပြီး စမ်းသပ်သည်။ Live ကို နောက်ဆုံးမှသာ ဖွင့်သည်။

---

## ၀။ အရင်ဆုံး နားလည်ရမယ့် architecture အချက် (plan တစ်ခုလုံးရဲ့ ကျောရိုး)

လက်ရှိ bot က **one-shot** ဖြစ်သည် — command တစ်ခု run တိုင်း `connect → execute → disconnect`
လုပ်ပြီး ထွက်သွားသည် (`main.py` ရှိ command function တိုင်း ဤပုံစံ; long-running loop **မရှိ**)။

ဒါက safety အတွက် အဓိကအကျဆုံး သက်ရောက်မှု ၂ ခု ရှိသည်:

1. **Software-side ထိန်းချုပ်မှု အားလုံး** (`loss_breached()`, hard-stop check, market-hours gate)
   က **bot run နေတဲ့ အချိန်မှာသာ** အလုပ်လုပ်နိုင်သည်။ Run နှစ်ခုကြားမှာ position တွေကို
   **ဘာမှ မစောင့်ကြည့်ပါ**။
2. ထို့ကြောင့် protective exit တွေက **broker-side resting order (GTC)** အဖြစ် ဖြစ်ကိုဖြစ်ရမည် —
   software check တင် မလုံလောက်ပါ။ Bot ပိတ်ထား/crash ဖြစ်နေချိန် gap down ဖြစ်ရင်၊ server မှာ
   resting stop မရှိရင် position က ကာကွယ်မှု လုံးဝမဲ့သွားမည်။

**ဆုံးဖြတ်ရမယ့် architecture fork (Phase 0 မှာ ဆုံးဖြတ်):**

| | Path A — Server-side + Scheduler *(အကြံပြု)* | Path B — Long-running Daemon |
|---|---|---|
| ပုံစံ | one-shot အတိုင်း ထား၊ position တိုင်းကို GTC stop နဲ့ ကာ၊ market-hours scheduler က entry/manage လုပ် | supervised process တစ်ခု continuous loop နဲ့ စောင့်ကြည့် + manage |
| အားသာချက် | ရိုးရှင်း၊ failure mode နည်း၊ bot down ဖြစ်လည်း stop က server မှာ ရှိ | real-time react, intraday logic ပိုလုပ်နိုင် |
| အားနည်းချက် | intraday react မရ (server stop ကသာ) | reconnect/watchdog/heartbeat အကုန်လို၊ failure mode များ |
| beginner သင့်လျော်မှု | ✅ မြင့် | ⚠️ နိမ့် |

➡️ **ဒီ plan က Path A ကို default အဖြစ် ယူသည်။** Path B လိုချင်ရင် Phase 5 မှာ daemon
အလွှာ ထပ်ဆောက်နိုင်သည် — ဒါပေမယ့် server-side GTC protection (Phase 3) က Path နှစ်ခုစလုံးအတွက်
မဖြစ်မနေ လိုအပ်သည်။

> ✅ **DECISION (2026-06-16):** **Path A** ကို ရွေးချယ်ပြီး — server-side GTC/OCA protection +
> market-hours scheduler။ နောက် phase အားလုံး ဒီ ဆုံးဖြတ်ချက်အပေါ် တည်ဆောက်သည်။
> Phase 0 အကောင်အထည်ဖော်မှု စတင်ပြီး (capability flag, audit log, invariant harness,
> `live-readiness` + `panic-flatten` command)。

---

## Phase ဆက်စပ်မှု (dependency) — ဘယ်အစဉ်နဲ့ လုပ်ရမလဲ

```
Phase 0 (foundation + test harness + path decision)
   │
   ▼
Phase 1 (truthful validation: C1, H20, H21, H23)   ──┐  (parallel-able)
Phase 2 (order execution robustness: H4–H9)        ──┤
   │                                                  │
   ▼                                                  │
Phase 3 (server-side protection + risk: C2,H1,H2,H3,H19) ◄┘  (needs Phase 2 fill-verify)
   │
   ▼
Phase 4 (data integrity: H11–H16)
   │
   ▼
Phase 5 (ops/monitoring/reconciliation: H17,H18,H10,M6)
   │
   ▼
Phase 6 (LIVE conversion — guard removal, account assertion, minimal capital) ◄ နောက်ဆုံး
```

Phase 1 နဲ့ Phase 2 က တစ်ခုနဲ့တစ်ခု မမှီခိုသဖြင့် တပြိုင်နက် လုပ်နိုင်သည်။ Phase 3 က Phase 2 ရဲ့
fill-verification အပေါ် မှီခိုသည် (filled qty/avgFillPrice မသိဘဲ မှန်ကန်တဲ့ stop မချနိုင်)။

---

## Phase 0 — အခြေခံ + Safety Test Harness (behavior မပြောင်းသေး)

**ရည်ရွယ်ချက်:** နောက် phase တိုင်းကို စစ်ဆေးနိုင်မယ့် paper-mode test အခြေခံ ဆောက်၊ architecture
path ရွေး။

| တာဝန် | ဘယ်မှာ | အသေးစိတ် |
|------|--------|----------|
| 0.1 Path A/B ဆုံးဖြတ် | — | အပေါ်က fork ကို ဆုံးဖြတ် (Path A အကြံပြု)၊ ဒီ plan ထဲ မှတ်တမ်းတင် |
| 0.2 Order-lifecycle audit log | `ibkr_bridge.py` | order တိုင်းရဲ့ submit→ack→fill→stop-placed→stop-confirmed အဆင့်တိုင်းကို structured (JSON) log; နောက် phase တွေ ဒီ log နဲ့ assert လုပ်မယ် |
| 0.3 Paper integration harness | `test_live_readiness.py` (အသစ်) | paper account နဲ့ end-to-end order path ကို run၊ invariant assert (အောက်တွင်) |
| 0.4 `live-readiness` self-check command | `main.py` | invariant အကုန် run၊ pass/fail report ထုတ်; live မသွားခင် ဒီ command က ✅ ဖြစ်ရမယ် |
| 0.5 `panic-flatten` command | `main.py` (`cancel_open_orders.py`/`flatten_vti.py` ပုံစံ) | position အကုန် market-close + working order အကုန် cancel; emergency kill-switch |

**Invariant များ (harness က assert လုပ်ရမယ့်):**
- open long တိုင်းမှာ live GTC protective stop ရှိရမယ်။
- `record_trade()` က actual fill ပြီးမှသာ ခေါ်ရမယ် (accepted တင် မဟုတ်)။
- child/exit qty = parent ရဲ့ actual filled qty။
- duplicate order မချဖြစ်ရ (restart simulation နဲ့ စစ်)။

**🚦 Go/No-Go gate:** harness run လို့ရ၊ `live-readiness` command က လက်ရှိ gap တွေကို
**မှန်ကန်စွာ FAIL** ပြနိုင်ရမယ် (baseline)။

---

## Phase 1 — Validation မှန်ကန်အောင် (C1, H20, H21, H23)

**ရည်ရွယ်ချက်:** "backtest မှာ မြင်ရတဲ့ edge" က "live မှာ ရမယ့် edge" နဲ့ တကယ် ကိုက်အောင်။

| တာဝန် | ဘယ်မှာ | အသေးစိတ် |
|------|--------|----------|
| 1.1 Production-parity backtest | `backtest.py` | `--include-lstm` နဲ့ run (C1)၊ ဖြစ်နိုင်ရင် fold-per-RF အစား **production final-refit RF** (`ai_engine.py:145-154` ရဲ့ saved model) ကို load-and-score mode ထည့်၊ live ensemble နဲ့ ၁:၁ ကိုက်အောင် |
| 1.2 Horizon ကိုက်ညှိ (H23) | `config.py`, `data_manager.py:157-161` | option A: exit ကို 5-day ATR-scaled stop ပြောင်း; option B: label ကို 1-day horizon ပြောင်း။ ပြီးမှ backtest ပြန် run |
| 1.3 Model metrics persist (H20) | `ai_engine.py:157-190` | train ရဲ့ oob/test acc/precision/recall/f1 + train-date ကို model file ဘေးမှာ persist (`models/model_metrics.json`) |
| 1.4 Accuracy floor + staleness gate | `predictor.py:225-275` | `predict_all` မှာ per-symbol metrics ဖတ်၊ accuracy floor အောက် သို့ max-age ကျော် symbol ကို **HOLD force** (H20, H21) |
| 1.5 Retrain scheduler hook | `model_doctor.py` | staleness ကို "report only" ကနေ "scheduled retrain trigger" အဖြစ် ချိတ် (Phase 5 scheduler နဲ့ ပေါင်း) |

**🚦 Go/No-Go gate:**
- `--include-lstm` backtest metrics က default backtest နဲ့ သိသိသာသာ မကွာ (signal frequency/edge)။
- Forward **paper** test အနည်းဆုံး **၁–၃ လ** run၊ live ensemble ရဲ့ trade frequency + win-rate က
  backtest expectation ရဲ့ tolerance အတွင်း ရှိရမယ်။ (မကိုက်ရင် Phase 6 မသွားရ။)

---

## Phase 2 — Order Execution ခိုင်မာအောင် (H4–H9) — အကြီးဆုံး engineering block

**ရည်ရွယ်ချက်:** "accepted" ဆိုတာ "filled" မဟုတ်။ Fill-driven, partial-aware, verified order layer
ဆောက်။

| တာဝန် | ဘယ်မှာ | အသေးစိတ် |
|------|--------|----------|
| 2.1 Fill-driven confirmation | `ibkr_bridge.py:289-306,308-350,352-386` | `ib.sleep(1)` (`:298,343,379`) အစား `trade.filledEvent`/`trade.isDone()` ကို timeout loop (`ib.waitOnUpdate`) နဲ့ စောင့်; outcome ကို `FILLED / WORKING / REJECTED / TIMEOUT` အဖြစ် explicit ပြန် |
| 2.2 "accepted" vs "filled" ခွဲ (H4) | `ibkr_bridge.py:27,36-40` | `ACCEPTED_ORDER_STATUSES` ကို order acceptance အတွက်သာ သုံး; **fill** ကို `orderStatus.status=='Filled'` + `remaining==0` နဲ့သာ သတ်မှတ် |
| 2.3 record_trade ကို fill ပေါ်ရွှေ့ | `ibkr_bridge.py:434-438,463-467` | `record_trade()` ကို **actual fill** ဖြစ်မှသာ ခေါ် (accepted တင် မဟုတ်) |
| 2.4 Partial-fill handling (H6) | `ibkr_bridge.py:331,378,442` | `orderStatus.filled/remaining/avgFillPrice` ဖတ်; child/exit qty = **actual filled qty**; `avgFillPrice` ကို stop basis အဖြစ်သုံး |
| 2.5 Protective-child verification (H7) | `ibkr_bridge.py:308-350` | parent fill ပြီးမှ child stop က `openTrades()` ထဲ correct `parentId` + working status ဖြစ်ကြောင်း confirm; မဖြစ်ရင် **ချက်ချင်း flatten + alert** |
| 2.6 Close-order robustness (H8) | `ibkr_bridge.py:289-306,419-420,441-443` | close ကို marketable-limit/market သုံး; `remaining==0` အထိ confirm + retry/escalate; bare limit + single check ဖျက် |
| 2.7 Idempotency + single-instance lock (H9) | `ibkr_bridge.py:107-154`, `risk_state.py` | deterministic `orderRef = f'{date}:{symbol}:{action}'` persist; startup မှာ `reqAllOpenOrders` + executions နဲ့ reconcile; process lock file |

**🚦 Go/No-Go gate (paper, harness နဲ့):**
- partial-fill simulate → child/exit qty က filled qty နဲ့ တိတိ ကိုက်။
- order reject simulate → bot က `REJECTED` အဖြစ် မှန်မှန် ကိုင်တွယ်၊ `record_trade()` မခေါ်။
- restart mid-flight simulate → duplicate order **မချ**။

---

## Phase 3 — Server-side Protection + Risk Engine (C2, H1, H2, H3, H19)

**ရည်ရွယ်ချက်:** position တိုင်း **server မှာ resting GTC stop** နဲ့ ကာ၊ ပိုက်ဆံ-based limit တွေ
တကယ် enforce ဖြစ်အောင်။ (Architecture keystone — Phase 0 ၏ အကြောင်းရင်း)

| တာဝန် | ဘယ်မှာ | အသေးစိတ် |
|------|--------|----------|
| 3.1 Hard stop ကို server-side GTC order အဖြစ် (C2) | `ibkr_bridge.py` | entry တိုင်းမှာ `HARD_STOP_LOSS_PCT` (-3%) ကို **independent server-side GTC STOP order** အဖြစ်ချ; software check မဟုတ် (one-shot ဖြစ်လို့ resting order ဖြစ်ကိုဖြစ်ရမယ်) |
| 3.2 GTC + OCA (H19) | `ibkr_bridge.py:323-342,370-378` | protective stop/trailing/TP အားလုံး `tif='GTC'`; SL+TP ကို explicit **OCA group** ထဲ (တစ်ခု fill ရင် ကျန်တာ auto-cancel)。 `protect_vti_gtc.py:38-42` ကို reference pattern အဖြစ်သုံး |
| 3.3 ATR-scaled stop (H2) | `config.py:158,171`, `ibkr_bridge.py:271-279` | fixed 0.4% အစား ATR% (တွက်ပြီးသား) အပေါ်အခြေခံ; spread+slippage ထက် မြင့်တဲ့ floor |
| 3.4 Daily-loss kill-switch wire (H1) | `risk_state.py:58-60`, `ibkr_bridge.py:389`, `main.py` | connect-time start-of-day equity (`NetLiquidation`) snapshot + persist; entry တိုင်းမတိုင်ခင် `loss_breached(start, current)` စစ်; breach ရင် entry အသစ်ပိတ် + alert (restart-safe) |
| 3.5 Account drawdown halt + exposure cap (H3) | `config.py`, `ibkr_bridge.py:469-517` (`execute_all`) | account-level max-drawdown halt; per-symbol/sector exposure cap; `FULL_MARKET_CORE_SYMBOLS` (semis/tech concentration) အတွက် correlation guard |
| 3.6 Startup protection invariant | `main.py`, `ibkr_bridge.py` | startup တိုင်း open long တိုင်းမှာ live GTC stop ရှိမရှိ scan; မရှိရင် **halt + alert + auto-repair** (stop ပြန်ချ) |

**🚦 Go/No-Go gate (paper):**
- paper မှာ position ဖွင့်ပြီး bot ပိတ်လိုက် → IBKR မှာ GTC stop **ကျန်နေရမယ်** (TWS မှာ မျက်မြင်စစ်)。
- start-of-day equity snapshot → loss simulate → entry အသစ် **ပိတ်ကြောင်း** စစ်。
- open long ၁ ခုကို stop မပါဘဲ ဖန်တီး → startup scan က **detect + repair** လုပ်ကြောင်း စစ်。

---

## Phase 4 — Data Integrity (H11–H16)

**ရည်ရွယ်ချက်:** order ချတဲ့ price က real-time + sane ဖြစ်ရမယ်; delayed/stale/yesterday price နဲ့
order မချရ။

| တာဝန် | ဘယ်မှာ | အသေးစိတ် |
|------|--------|----------|
| 4.1 Real-time data require (H12) | `config.py:215`, `ibkr_bridge.py:73` | live mode မှာ `IBKR_MARKET_DATA_TYPE=1` (real-time) require; subscription မရှိရင် order **refuse** |
| 4.2 Delayed/close field reject (H11,H13) | `ibkr_bridge.py:188-214` | live mode မှာ `close`/`delayedClose`/`delayedLast` field များ ဖယ်; real-time last → bid/ask midpoint သာ |
| 4.3 Decision price = order price (H14) | `predictor.py:256`, `trade_coach.py:270,307-316` | order ချမယ့် real-time IBKR quote ကို decision/preview မှာ သုံး; `|signal.price − live_price| > 0.5%` ဆို **refuse** |
| 4.4 Market-hours gate (H15) | `main.py` (order path), `ibkr_bridge.py` | `pandas_market_calendars` သို့ IBKR trading hours နဲ့ hard gate; RTH ပြင်ပ order refuse |
| 4.5 Bad-tick/spread sanity (H16) | `ibkr_bridge.py:179-186` | `(ask−bid)/mid > 1–2%` reject; `bid>=ask` (crossed) reject; recent price ±3–5% band check |
| 4.6 Historical fallback default ပိတ် (M5) | `ibkr_bridge.py:169` | `get_price(..., allow_historical=True)` default ကို `False` ပြောင်း (defense-in-depth; ယခု caller ပေါ်မူတည်) |

**🚦 Go/No-Go gate (paper):**
- delayed feed နဲ့ live mode → order **refuse** ဖြစ်ရမယ်。
- weekend/after-hours run → market-hours gate က **block** လုပ်ရမယ်。
- crossed/wide spread inject → order **reject** ဖြစ်ရမယ်。

---

## Phase 5 — Ops, Monitoring, Reconciliation (H17, H18, H10, M6)

**ရည်ရွယ်ချက်:** automated, supervised, broker-truth-based လည်ပတ်မှု + alert။

| တာဝန် | ဘယ်မှာ | အသေးစိတ် |
|------|--------|----------|
| 5.1 Scheduler (Path A) ✅ **(Phase 5B-1 done)** | `scheduler_runner.py` (pure decision), `main.py` `run-scheduled` command, `run_scheduled.bat` | one-shot supervised runner — paper-safety + `SCHEDULER_ENABLED` + market-hours gate (data_integrity calendar, fail-closed) စစ်ပြီးမှ ရှိပြီးသား `paper` command ကို ခေါ်; **daemon loop မရှိ**, **live order မချ**, gate တစ်ခုမှ မ bypass (model/data-integrity/market-hours/daily-loss/startup-reconcile/paper-port); plan/dry-run default (order မချ); `--execute` က `SCHEDULER_DRY_RUN_DEFAULT=False` ဖြစ်မှသာ paper order path သို့ (paper-locked အတိုင်း); decision ကို `order_audit` (`schedule` stage) မှာ မှတ်; `PYTHONUTF8=1` (`run_scheduled.bat`) UnicodeEncodeError ပြင်; offline tests `test_scheduler_runner.py`; **capability flag အသစ် မထည့်**. **watchdog (5B-2) ✅ / graceful shutdown (5B-3) ✅ / alerting (5B-4) ✅ — Phase 5B အကုန်ပြီး** |
| 5.2 Startup reconciliation (H18) ✅ **(Phase 5A done)** | `reconciliation.py` (pure), `ibkr_bridge.py` `reconcile_startup_state()`, `main.py` startup | startup တိုင်း `positions()` + `reqAllOpenOrders()` နဲ့ broker-truth snapshot rebuild; **broker = source of truth** (local file truth မယူ); protected/unprotected long + duplicate `orderRef` + orphan exit scan; unprotected long ကို Phase-3 `ensure_protective_stops` နဲ့သာ repair (entry အသစ် **မဖွင့်**); audit log (`reconcile` stage); offline tests `test_reconciliation.py`; `SUPPORTS_STARTUP_RECONCILIATION=True`. **executions()-based fill reconcile + alert delivery က Phase 5B** |
| 5.3 Reconnect/watchdog (H10) ✅ **(Phase 5B-2 done)** | `reconnect_watchdog.py` (pure backoff + `ConnectionHealth`), `ibkr_bridge.py` `connect()`/`disconnect()`, `config.py` `IBKR_RECONNECT_*` / `IBKR_REQUEST_TIMEOUT_SECONDS` | one-shot connection hardening — **daemon loop မရှိ**, forever retry မရှိ; paper-port lock ကို retry **မတိုင်ခင်** အရင်စစ် (bypass မဖြစ်); `IBKR_RECONNECT_ENABLED` ဖွင့်မှ initial connect ကို **bounded** exponential backoff (`MAX_ATTEMPTS` ပြည့်ရင် ရပ်) နဲ့ retry; `ib.RequestTimeout`/connect timeout set (`flatten_vti.py` ပုံစံ); `ib.disconnectedEvent` handler က မမျှော်လင့်ဘဲ disconnect ဖြစ်ရင် connection ကို **unhealthy** မှတ်ပြီး `_new_entries_blocked` က `connection_unhealthy` (fail-closed — entry အသစ်မဖွင့်), intentional `disconnect()` ကတော့ clean shutdown; **order လုံးဝ မချ**; `order_audit` (`watchdog` stage); offline tests `test_reconnect_watchdog.py`; **capability flag အသစ် မထည့်** (live-readiness NOT READY အတိုင်း). **graceful shutdown (5B-3) ✅ / alerting (5B-4) ✅ — Phase 5B အကုန်ပြီး** |
| 5.4 Graceful shutdown (M6) ✅ **(Phase 5B-3 done)** | `shutdown_guard.py` (pure planner), `ibkr_bridge.py` `prepare_graceful_shutdown()`/`graceful_shutdown()` | one-shot graceful teardown — resting protective SELL stop အားလုံး **ကျန်နေအောင်** preserve (`orders_to_cancel` အမြဲ `[]`), entry အသစ်မဖွင့်, unprotected long ကိုသာ **detect**; unprotected ဖြစ်ရင် ရှိပြီးသား Phase-3 `ensure_protective_stops` နဲ့သာ repair (dropped link မှာ fail-closed); disconnect ကို intentional မှတ်လို့ 5B-2 watchdog က clean close လို့ဖတ်; **order လုံးဝ မချ/မ cancel**; offline tests `test_shutdown_guard.py` |
| 5.5 Alerting ✅ **(Phase 5B-4 done)** | အသစ် `alerts.py`, `order_audit.py` (`alert` stage), `config.py` `ALERTS_*`, light wiring (`reconnect_watchdog.py` / `scheduler_runner.py` / `ibkr_bridge.py`) | disconnect / reconnect-failure / startup unprotected-long / orphan-exit / duplicate-orderRef / daily-loss kill-switch / order-reject / partial-fill / protective-child-failure(emergency-flatten) / scheduler-blocked / graceful-shutdown unprotected-long အတွက် operator alert။ **default disabled** (`ALERTS_ENABLED=False`), enable လုပ်လည်း **log-only** (`ALERTS_LOG_ONLY=True` → logger + `order_audit` `alert` stage သာ; box ပြင်ပ ဘာမှ မထွက်)။ `ALERT_MIN_SEVERITY` (default `warning`) severity filter။ `emit()` က trading path ထဲ **raise မဖြစ်**, **order မချ**, **live မဖွင့်**, flatten/repair ကို **မ block**။ email/SMS/Telegram/webhook က **disabled stub** (ဘာမှ မပို့)။ capability flag **အသစ် မထည့်**; offline tests `test_alerts.py`။ **❗ real channel မထည့်ရသေး — later phase မှ deliberately configure** |
| 5.6 (Path B ရွေးရင်) daemon loop | အသစ် | continuous monitor loop + heartbeat; 5.3 watchdog အပေါ်တည်ဆောက် |

**🚦 Go/No-Go gate (paper):**
- crash + restart → trade count + position state က **broker နဲ့ ကိုက်** (file count မဟုတ်)。
- network drop simulate → reconnect + alert ဖြစ်ရမယ်。
- unprotected position inject → alert + repair။

---

## Phase 6 — LIVE Conversion (နောက်ဆုံး — guard ဖြုတ်ခြင်း)

**⚠️ Phase 1–5 အကုန် go-gate အောင်ပြီးမှသာ ဒီ phase ကို ဝင်ရ။**

| တာဝန် | ဘယ်မှာ | အသေးစိတ် |
|------|--------|----------|
| 6.1 Account-type assertion **(အရေးအကြီးဆုံး guard gap)** | `ibkr_bridge.py:53-85`, `trade_coach.py:56-87` | connect ပြီး `managedAccounts()` ဖတ်; paper=`DU` prefix, live=`U` + explicit `LIVE_ACCOUNT_ID` match စစ်。 **Port lock က port ကိုသာ ကာ — wrong account ကို မကာ**; Gateway က 7497 မှာ live account login ထားရင် guard အကုန် ဖြတ်ပြီး real-money order ချနိုင်တယ် |
| 6.2 Live port wiring | `config.py`, `ibkr_bridge.py` | 7496 + client-id + account config **အသစ်ဆောက်** (flag ပြောင်းရုံ မဟုတ်) |
| 6.3 `live_safety_preflight()` | `main.py` | live order မချခင် invariant အကုန် (real-time data, market-hours, GTC stop capability, account-type, loss-limit snapshot) စစ်; တစ်ခုမှ fail ရင် refuse |
| 6.4 Config flag ပြောင်း | `config.py:175,184,202,258` | `COACH_LIVE_TRADING_ENABLED`, `REQUIRE_PAPER_PORT`, port — **အပေါ်အဆင့်အားလုံး ✅ ပြီးမှသာ** |
| 6.5 Minimal-capital rollout | — | share ၁ ခု/အသေးငယ်ဆုံး size, symbol ၁ ခု, supervised; ရက်အနည်းငယ် စောင့်ကြည့်ပြီးမှ တစ်ဆင့်ချင်း scale |

**🚦 Final Go-Live checklist:**
- [ ] `live-readiness` command ✅ (invariant အကုန်)
- [ ] Forward paper-test ၁–၃ လ, backtest expectation နဲ့ ကိုက်
- [ ] open position တိုင်း server-side GTC stop ရှိ (TWS မျက်မြင်)
- [ ] daily-loss kill-switch + drawdown halt တကယ် fire (paper မှာ စစ်ပြီး)
- [ ] account-type assertion ✅
- [ ] `panic-flatten` command ✅
- [ ] alerting ✅
- [ ] minimal capital + supervised

---

## ❌ မလုပ်ရ (Phase ၆ မရောက်မချင်း)

- `COACH_LIVE_TRADING_ENABLED = True` **မပြောင်းရ**
- `IBKR_PORT` ကို 7496 **မပြောင်းရ**, `REQUIRE_PAPER_PORT = False` **မလုပ်ရ**
- `ALLOW_HISTORICAL_PRICE_FOR_ORDERS = True` **မလုပ်ရ**
- account-type assertion မပါဘဲ live port connect **မလုပ်ရ**
- protective GTC stop logic (Phase 3) မပြီးဘဲ real-money order **မချရ**

---

## အကျဉ်းချုပ် — gate sequence

```
Phase 0 ✅ → (Phase 1 ✅ ∥ Phase 2 ✅) → Phase 3 ✅ → Phase 4 ✅ → Phase 5 ✅ → Phase 6 ✅ → LIVE (minimal capital)
```

တစ်ဆင့်စီ၏ go/no-go gate မအောင်ဘဲ နောက်တစ်ဆင့် **မသွားရ**။ ဒီ plan တစ်ခုလုံးကို **paper account**
ပေါ်မှာ ဆောက်ပြီး စမ်းသပ်သည်။ Live ကို Phase ၆ ၏ checklist အကုန် ✅ ဖြစ်မှသာ၊ အသေးငယ်ဆုံး
capital နဲ့ စတင်သည်။

---
*Source: verified findings — `reports/LIVE_TRADING_READINESS_MM.md` (CONFIRMED ၃၀/PARTIALLY ၁၃/FALSE ၁;
citation re-verify 2026-06-16, agent ၉ ခု)*
