# ─────────────────────────────────────────────────────────────────────────────
#  soxx_gate_config.py
#  SOXX Pre-Trade Gate — Enhanced Configuration
#  Enhancement 4: Relative Strength Bypass Layer
# ─────────────────────────────────────────────────────────────────────────────

# ── Original Gate Threshold (မပြောင်းပါ) ──────────────────────────────────────
SOXX_VETO_THRESHOLD          = -0.0075   # -0.75% — SOXX daily change floor

# ── Symbols Governed by SOXX Gate ─────────────────────────────────────────────
# ဤ list ထဲပါသော Symbols များသာ SOXX Gate ၏ ထိန်းချုပ်မှု အောက်တွင် ရှိသည်
SOXX_GOVERNED_SYMBOLS        = {"NVDA", "AMD", "AVGO", "SMCI", "INTC", "QCOM"}

# ── Relative Strength Bypass Criteria ─────────────────────────────────────────
# အောက်ပါ conditions ၂ ခုထဲမှ တစ်ခုခု ပြည့်မှီပါက SOXX Veto ကို Bypass ပြုလုပ်သည်

# Condition A — RSI Threshold
# Symbol ၏ RSI (14-period) သည် ဒီ value ထက် မနည်းမချင်း Bypass မပြုလုပ်ပါ
RS_BYPASS_RSI_FLOOR          = 55.0

# Condition B — Alpha vs SOXX
# Symbol ၏ intraday performance သည် SOXX ထက် ဒီ % ပိုကောင်းမချင်း Bypass မပြုလုပ်ပါ
RS_BYPASS_ALPHA_FLOOR        = 0.015     # +1.5% alpha vs SOXX

# Condition C — Minimum Predictor Confidence (AND condition)
# RS Bypass လုပ်ရန် Predictor Confidence အနည်းဆုံး ဒီ value ရှိရမည်
# (Weak signal + Strong RS = မယုံကြည်ရသော bypass ကို တားဆီးရန်)
RS_BYPASS_MIN_CONF           = 0.55

# ── Bypass Cooldown ────────────────────────────────────────────────────────────
# SOXX Gate ကို Bypass ပြုလုပ်ပြီးနောက် ဒီ scan cycle အရေအတွက် ပြည့်မချင်း
# ထပ်မံ Bypass မပြုလုပ်ပါ (Repeated aggressive entries ကာကွယ်ရန်)
RS_BYPASS_COOLDOWN_CYCLES    = 2

# ── Hard Veto Floor ────────────────────────────────────────────────────────────
# SOXX ဒီ value အောက် ကျဆင်းလျှင် RS Bypass ကို ဘယ်တော့မှ ခွင့်မပြုပါ
# (Sector Crash scenario — ကိုယ်ပိုင် RS မြင့်နေပေမယ့် Systemic Risk ကြောင့်)
SOXX_HARD_VETO_FLOOR         = -0.03    # -3.0% — absolute shutdown

# ── Lookback for RS Calculation ────────────────────────────────────────────────
RS_RSI_PERIOD                = 14       # RSI lookback periods
RS_ALPHA_LOOKBACK_BARS       = 1        # Intraday alpha: today's bar only
