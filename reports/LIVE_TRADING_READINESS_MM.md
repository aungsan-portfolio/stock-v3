> ## ⚠️ SUPERSEDED — ဒီ document က သမိုင်းမှတ်တမ်း (2026-06-16 baseline audit) သာ ဖြစ်သည်
> ဒီ readiness report က Phase 1–6 safety work **မလုပ်ရသေးခင်** ရေးထားတာ ဖြစ်လို့၊ အောက်ဖော်ပြ
> finding အများစုက **ယခု ပြင်ပြီးသွားပြီ** (2026-06-19 code audit + 484 tests pass ဖြင့် အတည်ပြုပြီး):
> - 🔴 **C2 hard stop (-3%)** — ယခု entry တိုင်းမှာ **server-side GTC STP** အဖြစ် ချပြီး (OCA group မျှ) ✅
> - ⚠️ **H1 daily-loss kill-switch** — ယခု `_new_entries_blocked` gate ထဲ **တကယ် wire ဖြစ်ပြီ** ✅
> - **H4–H9 order execution** (accepted≠filled, partial-fill, idempotency) ✅ · **H18 reconciliation** (broker-truth) ✅
> - **H15 market-hours gate** ✅ · **H19 GTC/OCA** ✅ · **account-type assertion** (Phase 6.1) ✅
>
> **လက်ရှိ source-of-truth = `reports/PAPER_GATE_VALIDATION_CHECKLIST.md`** (shipped Phase 0–5B+6.1 state)။
> ကျန်နေသေးတာ: Phase-1 forward paper-test (၁–၃ လ) + Phase 6.2/6.3 live wiring (PAPER ONLY ဆက်ထား)။
> အောက်က ၅၆-agent analysis ကို **roadmap/အကြောင်းရင်း မှတ်တမ်း** အဖြစ်သာ ဖတ်ပါ — current state အဖြစ် မဖတ်ပါနှင့်။

---

# Live Trading Bot ဖြစ်အောင် — အသေးစိတ် သုံးသပ်ချက် (မြန်မာ)

> Agent ၅၆ ခုနဲ့ codebase တစ်ခုလုံးကို subsystem ၆ ခု ခွဲဖတ် → dimension ၆ ခု audit →
> critical/high finding ၄၄ ခုကို adversarial verify လုပ်ပြီး ရရှိတဲ့ အကျဉ်းချုပ်။
> Verify ရလဒ်: **CONFIRMED ၃၀ / PARTIALLY_TRUE ၁၃ / FALSE ၁** — overstated finding တွေကို
> ပြန်ချိန်ညှိပြီး၊ တကယ်အရေးပါတဲ့အချက်တွေကိုပဲ ဖော်ပြထားသည်။

---

## ၁။ အကျဉ်းချုပ် — ဒီ bot က အခု ဘာလဲ

ဒီ project က **RF + LSTM + Technical ensemble → BUY/HOLD/SELL → IBKR** trading bot ဖြစ်ပြီး
**တမင်တကာ "paper-trading only" ဖြစ်အောင် ဒီဇိုင်းထုတ်ထားသည်**။ Live trading ကို guard အလွှာ
များစွာနဲ့ ပိတ်ဆို့ထားသည်။

**အရေးအကြီးဆုံး တကယ့်အချက် (သုံးသပ်ချက်ရဲ့ အနှစ်သာရ):**

ဒီ bot ကို live ဖြစ်အောင် "config flag ၅ ခု ပြောင်းရုံနဲ့" ပြောင်းလို့ **မရဘူး** —
ပြောင်းလို့ရပေမယ့် အဲဒါ **အလွန် အန္တရာယ်ကြီး** ပါတယ်။ ဘာကြောင့်လဲဆိုတော့ guard တွေက
**မရှိသေးတဲ့ live-trading safety logic ၁ စုံ** ကို နေရာယူ ဖုံးကွယ်ပေးနေတာ ဖြစ်လို့ပါ။
Paper မှာ "အလုပ်ဖြစ်နေတယ်" ထင်ရပေမယ့် — တကယ့်ပိုက်ဆံနဲ့ဆို ပေါက်ကွဲမယ့်အပေါက်တွေ
(stop မကပ်တဲ့ position, enforce မလုပ်တဲ့ loss limit, backtest နဲ့ live မတူတာ စသဖြင့်)
အများကြီး ကျန်နေသေးတယ်။

---

## ၂။ Paper-Only Guard များ — ဘယ်နေရာတွေက live ကို ပိတ်ထားလဲ

Live ဖြစ်အောင် ပြောင်းဖို့ ဒီ guard တွေ ဖြုတ်ရမယ်။ ဒါပေမယ့် **ဖြုတ်ရုံနဲ့ မလုံလောက်ဘူး**။

| # | Guard | တည်နေရာ | လ���ပ်ဆောင်ချက် |
|---|-------|---------|----------------|
| 1 | `assert_paper_trading_only()` | `trade_coach.py:56-88` | config flag ၅ ခုကို စစ်၊ တစ်ခုခု ဖောက်ရင် `LiveTradingDisabledError` |
| 2 | `REQUIRE_PAPER_PORT = True` | `config.py:202` | port 7497 မဟုတ်ရင် connect ပိတ် |
| 3 | port check | `ibkr_bridge.py:54-62` | `IBKR_PORT != PAPER_IBKR_PORT` ဆို connect refuse |
| 4 | `COACH_LIVE_TRADING_ENABLED = False` | `config.py:258` | master kill switch |
| 5 | `ALLOW_SHORT = False` | `config.py:175` | long-only |
| 6 | `ALLOW_HISTORICAL_PRICE_FOR_ORDERS = False` | `config.py:184` | order pricing အတွက် historical price ပိတ် |
| 7 | guard calls | `main.py:911, 1057` | bridge connect မလုပ်ခင် assert ၂ နေရာ |

**Live port (7496) wiring လုံးဝ မရှိ** — verify ဖြင့် အတည်ပြုပြီး (grep တွင် 7496 မတွေ့)။
Live-trading code path ကို **အသစ်ဆောက်ရမည်**။ Guard ဖြုတ်ရုံ မဟုတ်ပါ။

---

## ၃။ CRITICAL — ပိုက်ဆံမနဲ့ခင် မဖြစ်မနေ ပြင်ရမည် (Verify ဖြင့် CONFIRMED)

### 🔴 C1. Backtest က live နဲ့ မတူတဲ့ model ကို စစ်နေတယ် (LSTM ပျောက်နေ)
- **တည်နေရာ:** `config.py:194` (`BACKTEST_INCLUDE_LSTM=False`), `backtest.py:239-244`, `predictor.py:225-254`
- **ပြဿနာ:** Live signal က RF(0.40) + **LSTM(0.35)** + Tech(0.25) ၃-way blend။ ဒါပေမယ့်
  default backtest မှာ LSTM ကို ပိတ်ထား (`lstm_ok=False, score=0.5`) တဲ့အတွက် RF(0.40)+Tech(0.25)
  ပဲ စစ်နေတယ်။ **ဆုံးဖြတ်ချက်ရဲ့ ၃၅% (LSTM) က validation မှာ လုံးဝ မပါဘူး။**
  ထို့အပြင် backtest က fold တိုင်း RF အသစ်လေ့ကျင့်ပေမယ့်၊ live က history အကုန်နဲ့ refit လုပ်တဲ့
  RF တစ်ခုတည်း သုံးတယ် — ၂ ခု မတူ။
- **Live အန္တရာယ်:** `backtest_metrics.json` ကိုကြည့်ပြီး ပိုက်ဆံထည့်ဖို့ ဆုံးဖြတ်ပေမယ့်၊
  live ensemble ရဲ့ အပြုအမူ၊ signal frequency, confidence distribution တွေ မတူတဲ့အတွက်
  validated system က HOLD မယ့်နေရာမှာ live system က BUY လုပ်နိုင်တယ်။ Backtest ရဲ့
  edge/drawdown ကိန်းဂဏန်းတွေ **live ကို ကိုယ်စားမပြုဘူး။**
- **ဖြေရှင်းနည်း:** Go-live backtest ကို `include_lstm=True` နဲ့ run၊ ပြီးတော့ production ကို
  ship တဲ့ **တူညီတဲ့ final-refit RF** နဲ့ validate လုပ်ပါ။ Full-ensemble walk-forward နှေးရင်
  live ensemble ကို backtest လုပ်နိုင်တဲ့ ပုံစံအထိ လျှော့ပါ။

### 🔴 C2. Hard stop (-3%) ကို live မှာ လုံးဝ မသုံးဘူး — backtest ထဲမှာသာ ရှိ
- **တည်နေရာ:** `config.py:166` (`HARD_STOP_LOSS_PCT=0.03`), `predictor.py:180-189`,
  `backtest.py:213,261-266` (production code ထဲ **တစ်ခုတည်းသော** caller — `test_logic.py` မှာ
  unit test များသာ ခေါ်သုံး၊ live path က ဘယ်တုန်းကမှ မခေါ်), `ibkr_bridge.py:389-467`
- **ပြဿနာ:** `apply_position_rule_with_hold()` ထဲက -3% worst-case backstop ကို **backtest ကသာ**
  ခေါ်သုံးတယ်။ Live order path (`execute_signal`) က ဘယ်တုန်းကမှ မခေါ်ဘူး။ Live မှာ အကာအကွယ်က
  0.4% bracket stop တစ်ခုတည်းပဲ။ Code comment ကိုယ်တိုင်က "(conceptually) shared with live bridge"
  လို့ ရေးထား — "(conceptually)" ဆိုတဲ့ စကားလုံးက live မှာ မသုံးဘူးဆိုတာ ဖော်ပြနေတယ်။
- **Live အန္တရာယ်:** Backtest ရဲ့ equity curve, Sharpe, max-drawdown အားလုံးက မရှိတဲ့ -3% exit ကို
  ယူဆထားတယ်။ 0.4% stop က gap (earnings/news/halt) ကို ဖြတ်ကျသွားရင်၊ software backstop မရှိတဲ့အတွက်
  position တစ်ခုတည်းမှာ 10-30%+ ဆုံးရှုံးနိုင်တယ်။

> **မှတ်ချက်:** Verifier က C1, C2 ကို CRITICAL → ml dimension မှာ CRITICAL ဆိုပြီး အတည်ပြုထား။
> တခြား dimension တချို့မှာ ဒီ C2 ကိုပဲ HIGH လို့ ပြန်တွက်ထား (live entry stop က 0.4% ဖြစ်လို့
> သာမန်အခြေအနေမှာ ပိုစောစီးစွာ ထွက်တာကြောင့်)။ tail-risk ဖြစ်လို့ စာရင်းထဲ ဦးစားပေး ထားသင့်သည်။

---

## ၄။ HIGH — Live မဖြစ်ခင် ပြင်ရမည် (Verify ဖြင့် CONFIRMED အများစု)

### Risk Management (ပိုက်ဆံ ထိန်းချုပ်မှု)

**H1. Daily loss kill-switch ($150) က dead code — ဘယ်တုန်းကမှ မခေါ်ဘူး** ⚠️
- `risk_state.py:58-60` (`loss_breached`), `config.py:181` (`MAX_DAILY_LOSS_USD=150`)
- Grep တစ်ခုလုံးမှာ caller **သုည          ** — README မှာ ကြော်ငြာထားတဲ့ "$150/day drawdown halt"
  က runtime မှာ **လုံးဝ မရှိဘူး**။ တကယ် enforce ဖြစ်တာ trade **အရေအတွက်** cap
  (`can_open_more()`) တစ်ခုတည်းပဲ — ပိုက်ဆံ amount မဟုတ်ဘူး။
- **Live:** position ၃ ခု စလုံး stop ထိ၊ $150 ထက် ကျော်ဆုံးရှုံးနေတဲ့ကြားက bot က
  `MAX_DAILY_TRADES` ထိ entry အသစ်တွေ ဆက်ဖွင့်နေမယ်။ *(Verifier: bounded loss ဖြစ်လို့
  HIGH၊ unbounded မဟုတ်၊ ဒါပေမယ့် real gap-down မှာ ကျော်ဖောက်နိုင်တယ်။)*
- **ဖြေရှင်း:** connect မှာ start-of-day equity (NetLiquidation) snapshot ယူ၊ order တိုင်းမတိုင်ခင်
  current equity နဲ့ နှိုင်းပြီး breach ဖြစ်ရင် entry အသစ် ပိတ်၊ persist လုပ်ထား (restart-safe)။

**H2. Stop 0.4% က real slippage/spread ထက် ကျဉ်းလွန်း — whipsaw ဖြစ်မယ်**
- `config.py:158,171` (STOP=0.4%, TRAIL=0.4%), entry limit +0.1% offset
- Backtest ကိုယ်တိုင်က order တစ်ခုစီ 5bps slippage + 5bps cost (~0.1% round-trip) ယူဆ၊
  ဒါပေမယ့် live stop band က 0.4% ပဲ။ Liquid name အများစုရဲ့ သာမန် intraday wiggle နဲ့ bid/ask
  bounce က 0.4% ကို အလွယ်တကူ ထိတယ်။
- **Live:** entry +0.1% → -0.4% stop ထိ → round-trip cost ဆောင် → ထပ်လုပ်... =
  **slippage-and-commission bleed machine**။ Trailing 0.4% က real winner တွေကိုလည်း စောစော ဖြတ်တယ်။
- **ဖြေရှင်း:** ATR% (တွက်ပြီးသား ရှိ) အပေါ် အခြေခံတဲ့ volatility-scaled stop သုံး၊
  spread+slippage ထက် ကောင်းကောင်းမြင့်တဲ့ floor ထား။

**H3. Portfolio-level control မရှိ — MAX_OPEN_POSITIONS=3 က correlated concentration ခွင့်ပြု**
- `config.py:155`, `config.py:48-52` (core symbols = NVDA/AMD/AVGO/MSFT/META — semis/tech စု)
- Max-drawdown halt မရှိ၊ per-symbol/sector exposure cap မရှိ၊ correlation check မရှိ။
  "tiny positions အနည်းငယ်" ဆိုတဲ့ ရည်ရွယ်ချက်က illusory — open slot ၃ ခုစလုံး correlated
  AI/semis ဖြစ်နိုင်လို့ တကယ်တော့ concentrated position ၁ ခုပဲ။
- **Live:** sector down-day တစ်ရက်တည်းမှာ ၃ ခုစလုံး stop တစ်ပြိုင်နက် ထိမယ်။
- **ဖြေရှင်း:** account drawdown halt, per-symbol/sector cap, correlation check ထည့်။

### Order Execution (order ချမှု — ဒီ dimension က CONFIRMED အများဆုံး)

**H4. "Accepted" ကို "success" လို့ မှားယူ — PreSubmitted ကို ပြီးပြည့်စုံတဲ့ trade လို့ မှတ်**
- `ibkr_bridge.py:27,36-40,300-306,434-438`
- `ACCEPTED_ORDER_STATUSES` ထဲ Filled မဟုတ်တဲ့ PendingSubmit/PreSubmitted/Submitted ပါ။
  `ib.sleep(1)` ပြီး status ၁ ကြိမ်စစ်၊ acknowledge ဖြစ်ရုံနဲ့ `accepted=True` ပြန်ပြီး
  `record_trade()` ခေါ်တယ် — share ၁ ခုမှ မဖြည့်ရသေးဘဲ။
- **Live:** order က fill မဖြစ်ဘဲ "accepted" လို့ report ၊ daily-trade slot ၁ ခု ကုန်၊
  operator က position ဖွင့်ပြီးပြီထင် (တကယ်မဖွင့်ရသေး)၊ သို့မဟုတ် protective close အောင်မြင်ပြီထင်။
- **ဖြေရှင်း:** "accepted by API" နဲ့ "filled" ခွဲ။ `trade.filledEvent` / `trade.isDone()` နဲ့
  timeout အတွင်း Filled အထိ poll၊ Filled မှသာ `record_trade()`။

**H5. Fixed `ib.sleep(1)` က confirmation မဟုတ်၊ race ဖြစ်**
- `ibkr_bridge.py:298,343,379`
- ၁ စက္ကန့်က submission + exchange ack + fill အတွက် မလုံလောက်။ Rejected (buying power မလောက်/halt/
  price-band) တွေ t+1.5-3s မှ ရောက်လေ့ရှိ — bot က t+1s မှာ "accepted" မှတ်ပြီးသွားပြီ။
- **ဖြေရှင်း:** `ib.waitOnUpdate(timeout)` loop နဲ့ `trade.isDone()` အထိ စောင့်၊
  "still working at deadline" ကို explicit outcome အဖြစ် ကိုင်တွယ်။

**H6. Partial-fill handling မရှိ — filled qty ကို ဘယ်တုန်းကမှ မဖတ်**
- `ibkr_bridge.py:340-350,370-386,442-443`
- `orderStatus.filled/remaining/avgFillPrice` ကို ဘယ်နေရာမှ မဖတ်။ Parent က 30/100 share ပဲ
  fill ပေမယ့် child TRAIL ကို 100 share အတွက် ချ — exit oversized ဖြစ်ပြီး trigger ရင် **short
  ဖြစ်သွားနိုင်**။ Close-long က partial အတွက် follow-up မရှိ။
- **ဖြေရှင်း:** parent fill ပြီးမှ `orderStatus.filled` ဖတ်ပြီး child/exit ကို actual qty နဲ့ size၊
  `avgFillPrice` ကို stop/trailing basis အဖြစ်သုံး။

**H7. Trailing-stop child verification မရှိ — stop တကယ်ကပ်ပြီလားဆို မစစ်**
- `ibkr_bridge.py:308-350`
- Parent (transmit=False) + child TRAIL (transmit=True) ချ၊ sleep(1) ပြီး PreSubmitted ကို
  "accepted" မှတ်။ Child တကယ် live ဖြစ်ပြီလားဆို မစစ်ဘူး။
- **Live:** parent က millisecond အတွင်း fill ဖြစ်ပြီး child က PendingSubmit/Rejected ဖြစ်နေရင်
  position က **stop မကပ်ဘဲ naked** — bot က order ချပြီး disconnect လုပ်လို့ ဘာမှ မစောင့်ကြည့်တော့။
- **ဖြေရှင်း:** parent Fill ပြီးမှ child က openTrades() ထဲ correct parentId နဲ့ working status
  ဖြစ်ကြောင်း အတည်ပြု၊ မဟုတ်ရင် ချက်ချင်း flatten + alert။

**H8. Close order တွေက bare limit — stop မရှိ၊ fill verification မရှိ**
- `ibkr_bridge.py:289-306,419-420,441-443`
- Position ပိတ်တဲ့အခါ price*(1±0.1%) limit ၁ ခုပဲ ချ၊ t+1s status ၁ ကြိမ်စစ်။
- **Live:** ကျနေတဲ့ စျေးကွက်မှာ -0.1% sell-to-close limit က fill မဖြစ်ဘဲ ကျန်နိုင်၊ bot က "closed"
  မှတ်ပေမယ့် သင်က stop မရှိဘဲ long ကျန်နေတယ်။ Retry/escalate မရှိ။
- **ဖြေရှင်း:** close အတွက် marketable-limit/market သုံး၊ remaining==0 အထိ confirm + retry။

**H9. Idempotency က timing-sensitive snapshot အပေါ်တည် — restart မှာ double-place/skip**
- `ibkr_bridge.py:107-154`, `risk_state.py:49-55`
- Duplicate prevention က `has_working_order()` (reqAllOpenOrders + sleep(1)) တစ်ခုတည်း။
  Client-side order key/dedup မရှိ၊ intended orders persist မလုပ်။ 1s အတွင်း populate မဖြစ်ရင်
  bot က duplicate ချမယ်။
- **ဖြေရှင်း:** `orderRef = f'{date}:{symbol}:{action}'` deterministic key persist လုပ်၊
  startup မှာ reqAllOpenOrders + executions နဲ့ reconcile။

**H10. Mid-order disconnect handling မရှိ** *(PARTIALLY_TRUE — order-exec context မှာ HIGH)*
- `ibkr_bridge.py:53-85`
- connect() က exception ဖမ်းပြီး False ပြန်ရုံ (retry/backoff မရှိ)၊ `disconnectedEvent` handler
  မရှိ၊ `ib.RequestTimeout` ဒီ bridge မှာ မ set (flatten_vti.py မှာတော့ set ထား)။

**H11. Snapshot price က stale — limit နဲ့ qty ကို မှားတွက်စေ**
- `ibkr_bridge.py:169-230`, `config.py:215` (delayed type 3)
- get_price() က last/close/delayedLast/delayedClose ပြီးမှ bid/ask midpoint — 15-min delayed
  feed default။ stale price က limit (±0.1%) နဲ့ qty (`_calc_quantity`) ၂ ခုစလုံးကို မှားစေတယ်။

### Data Integrity (ဒေတာ မှန်ကန်မှု / latency)

**H12. IBKR market data က DELAYED (15-min) hard-coded — live order pricing source**
- `config.py:215` (`IBKR_MARKET_DATA_TYPE=3`), `ibkr_bridge.py:73`
- **ဖြေရှင်း:** live မှာ type 1 (real-time) + subscription မရှိရင် order ချတာ refuse၊
  delayed field အားလုံး reject။

**H13. get_price() က live bid/ask အစား settled 'close' ကို ဦးစားပေး — မနေ့က စျေး ပြန်ပေး**
- `ibkr_bridge.py:189-214`
- After-hours/gap မှာ မနေ့က close နဲ့ size + bracket လုပ်တယ်။ 5% gap up ဆို limit က fill မဖြစ်၊
  gap down ဆို falling knife ထဲ ဝယ်မိမယ်။
- **ဖြေရှင်း:** live မှာ 'close'/'delayedClose' field တွေ ဖယ်၊ real-time last → bid/ask midpoint
  (max-spread check နဲ့)။

**H14. Signal price က yfinance EOD close — IBKR ကနေ မဟုတ်**
- `predictor.py:256`, `trade_coach.py:270,307-316`
- Operator က entry $100, stop $99.60, max loss $2 ထင်ပြီး confirm လုပ်ပေမယ့် bridge က IBKR price
  $103 (gap) နဲ့ fill — **ပြသထားတဲ့ risk က စိတ်ကူးယU
ဉ်**။ Weekend/holiday မှာ stale ပိုဆိုး။
- **ဖြေရှင်း:** decision/preview price ကို order ချမယ့် real-time IBKR quote ကနေ ယူ၊ သို့မဟုတ်
  `|signal.price - live_price| > 0.5%` ဆို refuse။

**H15. Market-hours / weekend / holiday awareness လုံးဝ မရှိ** *(grep ဖြင့် absent အတည်ပြု)*
- Saturday/holiday/3am မှာ run ရင် Friday close ကို "price" အဖြစ်သုံး။ RTH ပြင်ပ order တွေ
  open မှာ မမျှော်လင့်တဲ့ price နဲ့ fill ဖြစ်နိုင်။
- **ဖြေရှင်း:** `pandas_market_calendars` သို့ IBKR trading hours နဲ့ hard market-hours gate။

**H16. Bad-tick / spread / sanity validation မရှိ**
- `ibkr_bridge.py:179-214` (`_finite_positive` က >0 နဲ့ NaN မဟုတ်ရုံ စစ်)
- crossed/wide spread midpoint က qty + stop ကို မှားစေ။
- **ဖြေရှင်း:** `(ask-bid)/mid > 1-2%` reject, `bid>=ask` reject, recent price ±3-5% band check။

### Ops / Monitoring (လည်ပတ်မှု / စောင့်ကြည့်မှု)

**H17. Scheduler/automation မရှိ — `run_dryrun.bat` က dry-run သီးသန့်၊ run တိုင်း crash**
- `run_dryrun.bat:3`, `logs/scheduled_dryrun.log` (UnicodeEncodeError)
- "manual run" ဆိုတာ operator သတိရမှ trade — entry ရော **exit/stop ပါ** လူ keyboard ရှေ့ရှိမှ ချ။
- **ဖြေရှင်း:** market-hours scheduler (supervised long-running process သို့ OS scheduler) +
  `PYTHONUTF8=1` environment (bare `python main.py` မဟုတ်)။

**H18. Startup reconciliation မရှိ — file-backed trade count က account နဲ့ တိမ်းပါး**
- `risk_state.py:18-55`, `main.py:899-919`
- Broker fills/positions နဲ့ ဘယ်တုန်းကမှ reconcile မလုပ်။ crash + restart ဆို count မှား →
  daily cap ကို silently bypass။ "ကိုယ်ထင်တာ" နဲ့ "account မှာ တကယ်ရှိတာ" မညီ — crashed run က
  ဖွင့်ထားတဲ့ unprotected position ကို မသိ။
- **ဖြေရှင်း:** startup တိုင်း positions() + reqAllOpenOrders() + executions နဲ့ rebuild၊
  broker ကို source of truth အဖြစ်သုံး၊ unprotected-position scan + alert ထည့်။

**H19. Order TIF က default DAY, OCA group မရှိ — stop က session ကျော် မတည်**
- `ibkr_bridge.py:323-342,370-378` vs `protect_vti_gtc.py:38-42` (GTC+OCA တမင်set)
- DAY stop က close မှာ သေ။ နောက်နေ့ refresh မလုပ်ရင် (scheduler မရှိ) open long က **stop မကပ်ဘဲ**
  overnight သယ်သွား → gap-down ထိ။ `protect_vti_gtc.py` ရှိနေတာကိုက stop သက်တမ်းကုန်ဖူးတဲ့
  သက်သေ။
- **ဖြေရှင်း:** protective stop/trailing child မှာ `tif='GTC'` set၊ TP+SL ကို explicit OCA group ထဲ၊
  startup invariant: open long တိုင်း live GTC stop ရှိရမယ်၊ မရှိရင် halt + alert။

### ML / Signal Reliability (model ယုံကြည်စိတ်ချရမှု)

**H20. Minimum-performance gate မရှိ — 45% accuracy model က 80% model လို trade**
- `ai_engine.py:157-178`, `predictor.py:225-275`, `ibkr_bridge.py:434`
- Training က oob/test acc/precision/recall/f1 တွက်ပေမယ့် **log ထဲပဲ** — predict/execute path က
  ဘယ်တုန်းကမှ မဖတ်။ Accuracy floor မရှိ၊ per-symbol suspension မရှိ။ 5-day equity direction က
  coin-flip နီးပါး၊ depth-6 RF + small LSTM က 0.50 နီးပါး ဖြစ်တတ်။
- **ဖြေရှင်း:** per-symbol metrics (CV accuracy, val loss, train date) ကို model နဲ့အတူ persist၊
  predict_all မှာ accuracy floor အောက် symbol ကို HOLD force။

**H21. Model staleness က informational ပဲ — trade မ block, retrain scheduler မရှိ**
- `model_doctor.py:49-51,122-133` ("reporting hint only - never blocks trading")
- Regime shift (rate/vol/rotation) ဖြစ်ရင် model က ဟောင်းတဲ့ regime အတိုင်း full confidence နဲ့
  BUY ဆက်ထုတ်နေမယ်၊ ဘာမှ မသတိပေး။
- **ဖြေရှင်း:** per-symbol train timestamp store၊ max-age ကျော်ရင် HOLD force၊ scheduled retrain။

**H22. ML model ၁ ခု (LSTM ပျောက်) ဆို technical heuristic က BUY ဆုံးဖြတ်ချက် ~38% ဖြစ်သွား**
- `config.py:147` (`MIN_ML_MODELS_FOR_SIGNAL=1`), `predictor.py:105-125`
- LSTM ပျောက်ရင် weighted_blend renormalize → RSI/MACD/BB heuristic က executed decision ရဲ့ ~38%။
  Freshly-rotated symbols (trade ဖြစ်နိုင်ဆုံး) မှာ coverage ပါးလို့ single-ML BUY များတယ်။
- **ဖြေရှင်း:** live မှာ `MIN_ML_MODELS_FOR_SIGNAL=2` (RF + LSTM ၂ ခုလို) သို့ technical share cap။

**H23. 5-day forward label vs 0.4% exit — horizon mismatch**
- `config.py:114,138,158,171`, `data_manager.py:157-161`
- Model က "5 ရက်အတွင်း >0.3% တက်မယ်" ဟောဖို့ train။ ဒါပေမယ့် live exit က 0.4% trailing stop။
  တစ်ရက် noise (~1-2%) က 0.4% trailing ကို အလွယ်ထိ — trade အများစုက 5-day window မရောက်ခင်
  တစ်ရက်နှစ်ရက်အတွင်း stop out။ **Model ရဲ့ skill (5-day move) နဲ့ exit (1-day noise) က မတူ။**
- **ဖြေရှင်း:** trailing/stop ကို 5-day volatility (ATR-scaled) နဲ့ ကိုက်ညှိ၊ သို့ label ကို
  1-day horizon ပြောင်း၊ ပြီးမှ backtest ပြန် run။

---

## ၅။ MEDIUM — ဂရုစိုက်သင့်သော်လည်း blocker မဟုတ်

- **M1. order ချနိုင်တဲ့ path အားလုံးမှာ `assert_paper_trading_only()` မရှိ** — guard က
  `_daily_coach_execute` (`main.py:911`) နဲ့ `_run_daily_coach` (`main.py:1057`) ၂ နေရာသာ ဖြတ်တယ်။
  `cmd_paper_coach` (`main.py:625`) ရော `paper`/`paper-hot --execute` path တွေက guard မဖြတ်ဘဲ
  bridge ကို တိုက်ရိုက် connect လုပ်တယ်။ (PARTIALLY_TRUE: port lock + `COACH_LIVE_TRADING_ENABLED`
  က ဆက်ကာကွယ်နေဆဲ ဖြစ်လို့ MEDIUM — ဒါပေမယ့် guard ကို connect-time central choke-point တစ်ခုထဲ
  စုစည်းသင့်သည်)
- **M2. Daily trade counter က restart မှာ forget/reset** *(H9/H18 နဲ့ ဆက်စပ်)*
- **M3. Confidence blend က uncalibrated** — 0.65/0.40 threshold က probability မဟုတ်၊
  arbitrary weighted score။
- **M4. yfinance က price/feature source တစ်ခုတည်း** — unreliable, rate-limited, EOD-oriented။
- **M5. Historical-close fallback က config flag ၁ ခုနဲ့သာ ပိတ်** — defense-in-depth ပါး။
- **M6. Graceful shutdown / signal handler / top-level exception guard မရှိ** (PARTIALLY_TRUE →
  MEDIUM: transmit semantics + paper lock ကြောင့် naked-fill exploit က overstated, ဒါပေမယ့်
  SIGINT/SIGTERM/atexit guard မရှိတာ live အတွက် ပြင်သင့်)။

---

## ၆။ Verifier က overstated လို့ ဆုံးဖြတ်တာများ (မှန်ကန်မှု အတွက် ဖော်ပြ)

Adversarial verification ၏ တန်ဖိုး — အောက်ပါတို့ကို **down-grade/FALSE** လုပ်ခဲ့သည်:

- ❌ **FALSE:** "`assert_paper_trading_only()` က dimension ၅ ခု couple လုပ်ထားလို့ flag တစ်ခု
  မှားရင် silently live ဖြစ်" — guard ၅ ခုစလုံး ဖောက်မှ live ဖြစ်တာမို့ load-bearing claim မှား။
- ⬇️ **HIGH → LOW:** "connection one-shot, no reconnect/watchdog → mid-run disconnect silent &
  fatal" — long-running daemon **မရှိ**။ command တိုင်းက one-shot (connect → execute → disconnect)၊
  exit/stop တွေ server-side ဖြစ်လို့ "orphaned position" claim က current code နဲ့ မကိုက်။
- ⬇️ **HIGH → MEDIUM:** "no graceful shutdown → naked filled long" — `transmit=False` parent
  semantics က crash window ကို ပိတ်ပြီးသား (parent activate မဖြစ်ဘဲ ကျန် = fill မရှိ)၊ paper lock
  ပါရှိ။
- ⬇️ **HIGH → LOW:** "production RF refit on ALL history, zero holdout" — ai_engine က chronological
  holdout မှာ အရင် evaluate ပြီးမှ refit လုပ်တာ မှန်လို့ overstated။

---

## ၇။ Live Trading သို့ ပြောင်းရန် — အဆင့်ဆင့် Roadmap

> **ရည်ရွယ်ချက်:** guard ဖြုတ်တာ မဟုတ်ဘဲ၊ guard တွေ ဖုံးကွယ်ထားတဲ့ **မရှိသေးတဲ့ safety logic**
> ကို အရင်ဆောက်၊ ပြီးမှ guard ဖြုတ်။

### အဆင့် 0 — အခု ဆက်လုပ်ပါ (ပြောင်းစရာ မလို)
- Paper မှာ ဆက် run၊ `daily-coach` flow နဲ့ လေ့ကျင့်။ **ဒီ bot က အခု live အတွက် အသင့်မဖြစ်သေး။**

### အဆင့် 1 — Validation မှန်အောင် (C1)
1. Backtest ကို `--include-lstm` + production final-refit RF နဲ့ run၊ live ensemble နဲ့ ၁:၁ ကိုက်အောင်။
2. Forward paper-test အနည်းဆုံး ၁-၃ လ၊ backtest expectation နဲ့ နှိုင်း။

### အဆင့် 2 — Risk & Stop logic ပြည့်စုံအောင် (C2, H1, H2, H19)
3. `loss_breached()` ကို execution gate ထဲ wire (start-of-day equity snapshot + persist)။
4. Hard stop (-3%) ကို server-side GTC stop အဖြစ် live entry တိုင်းမှာ ချ (independent leg)။
5. Stop ကို ATR-scaled ပြောင်း (0.4% fixed မဟုတ်)၊ TIF=GTC + OCA group။
6. Max-drawdown halt + per-symbol/sector exposure cap (H3)။

### အဆင့် 3 — Order execution robustness (H4-H9)
7. "accepted" vs "filled" ခွဲ — `trade.filledEvent`/`isDone()` နဲ့ event-driven wait။
8. Partial-fill handling (filled qty ဖတ်ပြီး child size)။
9. Protective child verification (parent fill ပြီး stop live ဖြစ်ကြောင်း confirm, မဟုတ်ရင် flatten)။
10. Close order ကို marketable + retry + confirm။
11. Persisted `orderRef` idempotency + single-instance lock။

### အဆင့် 4 — Data integrity (H11-H16)
12. Real-time data (type 1) require၊ delayed field reject၊ EOD-close reject (live mode)။
13. Signal price ကို IBKR real-time quote ကနေ ယူ (yfinance မဟုတ်)။
14. Market-hours/weekend/holiday gate။
15. Bad-tick/spread sanity validation။

### အဆင့် 5 — Ops & monitoring (H17, H18, H10, M6)
16. Market-hours scheduler (supervised, UTF-8 env)။
17. Startup reconciliation (broker = source of truth) + unprotected-position scan + alert။
18. Reconnect/watchdog (`ib.disconnectedEvent` + backoff) + heartbeat (live daemon ဖြစ်လာရင်)။
19. SIGINT/SIGTERM/atexit graceful shutdown။
20. Alerting (email/SMS/push) on halt, disconnect, unprotected position, daily-loss breach။

### အဆင့် 6 — Conversion ကိုယ်တိုင် (guard ဖြုတ်ခြင်း — အနောက်ဆုံး)
21. **Account-type assertion ထည့်** (အရေးကြီး — H-level guard gap): connect ပြီး
    `managedAccounts()` ဖတ်ပြီး paper ဆို `DU` prefix, live ဆို `U` + explicit `LIVE_ACCOUNT_ID`
    နဲ့ ကိုက်အောင် စစ်။ Port lock က **port ကိုသာ** ကာကွယ်တယ်၊ wrong account ကို မကာ — Gateway က
    7497 မှာ live account login ထားရင် guard အကုန်ဖြတ်ပြီး real money order ချနိုင်တယ်။
22. Live port (7496) wiring + client-id + account config အသစ်ဆောက်။
23. config flag တွေ ပြောင်း — **အပေါ်အဆင့်အားလုံး ပြီးမှသာ**။
24. အသေးငယ်ဆုံး capital နဲ့ စ၊ တစ်ဆင့်ချင်း scale။

---

## ၈။ နိဂုံး

ဒီ bot က code quality အရ **သေသပ်ပြီး၊ paper-trading အတွက် safety-conscious** ဖြစ်ပါတယ် —
guard အလွှာများ၊ duplicate guard၊ bracket order၊ MIN_ML_MODELS_FOR_SIGNAL စတာတွေ တွေ့ရတယ်။

ဒါပေမယ့် **live trading အတွက် အသင့်မဖြစ်သေးပါ**။ အဓိကအချက်က — guard တွေက
*"လုပ်ထားပြီးသား safety logic"* ကို ပိတ်ထားတာ မဟုတ်ဘဲ၊ *"မရှိသေးတဲ့ safety logic"* ကို
ဖုံးကွယ်ပေးထားတာပါ။ confirmed gaps ၂၅+ (CRITICAL ၂ + HIGH ၂၃) ထဲမှာ —
**daily-loss kill-switch က dead code**, **hard stop က live မှာ မရှိ**, **backtest က live နဲ့ မတူ**,
**stop က session ကျော် မတည် (DAY TIF)**, **partial fill/account-type/market-hours/scheduler အကုန်
မရှိ** — ဒါတွေ မဖြေရှင်းဘဲ live သွားရင် ပိုက်ဆံ ဆုံးရှုံးဖို့ သေချာနီးပါးပါ။

**အကြံပြုချက်:** Roadmap အဆင့် 1-5 အကုန်ပြီးအောင် (အထူးသဖြင့် C1, C2, H1, H4-H9, H19, H21,
account-type assertion) လုပ်ပြီးမှသာ၊ အသေးငယ်ဆုံး capital နဲ့ live စမ်းသပ်သင့်ပါတယ်။

---
*Workflow: ၅၆ agents · subsystem ၆ + dimension ၆ + verified findings ၄၄ · CONFIRMED ၃၀/PARTIALLY ၁၃/FALSE ၁*

*Citation re-verify (2026-06-16): agent ၉ ခုနဲ့ finding တိုင်းရဲ့ file:line ကို လက်ရှိ code နဲ့ ပြန်တိုက်ဆိုင် —
အကုန်လုံးနီးပါး **CONFIRMED**။ ပြင်ဆင်ချက် အသေး: C2 hard-stop logic က `predictor.py:180-189`
(line 179 က comment)၊ production code ထဲ caller က `backtest.py` တစ်ခုတည်း (`test_logic.py` က test သာ)၊
guard မရှိတဲ့ တတိယ path `cmd_paper_coach` (`main.py:625`) ထပ်တွေ့လို့ M1 ကို ပြင်ဆင်ထား။*

> 📋 **အကောင်အထည်ဖော်မှု အဆင့်ဆင့် အသေးစိတ် plan:** `reports/LIVE_TRADING_IMPLEMENTATION_PLAN_MM.md` တွင်
> ဆက်ဖတ်ပါ — code-level တာဝန်များ၊ paper-test gate များ၊ go/no-go criteria များ ပါဝင်သည်။
