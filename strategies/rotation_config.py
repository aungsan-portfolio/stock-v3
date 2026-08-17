# ─────────────────────────────────────────────────────────────────────────────
#  rotation_config.py
#  Dynamic Position Rotation — Configuration Constants
# ─────────────────────────────────────────────────────────────────────────────

# ── Rotation Trigger Thresholds ───────────────────────────────────────────────

# Incoming signal ၏ Confidence သည် ဒီ value ထက် မများမချင်း Rotation မလုပ်ပါ
ROTATION_MIN_INCOMING_CONF   = 0.58

# Existing position ၏ current re-scored Confidence သည် ဒီ value အောက်ကျမှ
# Rotation Candidate အဖြစ် သတ်မှတ်ပါသည်
ROTATION_WEAK_POSITION_CONF  = 0.48

# Incoming conf နှင့် Existing conf ၏ ကွာဟချက် ဒီ value ထက် ကြီးမှ Rotate လုပ်ပါ
# (Churn ဖြစ်ကာ အကြောင်းမဲ့ trade မဖြစ်ရအောင်)
ROTATION_MIN_CONF_DELTA      = 0.10

# ── P&L Guard ────────────────────────────────────────────────────────────────

# Profit % ဒီ value ထက် ကြီးနေတဲ့ Winner ကို ဘယ်တော့မှ မဖယ်ထုတ်ပါ
# (Winners များကို Trend စီးမျောနိုင်ရန်)
ROTATION_PROTECT_PROFIT_PCT  = 0.08    # 8% profit သည် Protected Winner

# Loss % ဒီ value ထက် ကြီးနေတဲ့ Loser ကို Rotation မစောင့်ဘဲ Hard Stop မှ ကိုင်တွယ်ပါ
ROTATION_HARD_STOP_LOSS_PCT  = -0.07   # -7% သည် Hard Stop territory

# ── Cooldown ─────────────────────────────────────────────────────────────────

# တစ်ကြိမ် Rotate ပြီးလျှင် ဒီ scan cycle အရေအတွက် ပြည့်မချင်း
# ထပ်မံ Rotate မလုပ်ပါ (Flip-flopping ကာကွယ်ရန်)
ROTATION_COOLDOWN_CYCLES     = 3

# ── Max Rotations per Cycle ──────────────────────────────────────────────────

# တစ်ကြိမ် Scan Cycle တွင် တစ်ကြိမ်ထက်ပို Rotate မလုပ်ပါ
MAX_ROTATIONS_PER_CYCLE      = 1

# ── Score Weights (Rotation Decision Score) ──────────────────────────────────
# Rotation Score = (conf_weight × conf) + (pnl_weight × pnl_score)
SCORE_CONF_WEIGHT            = 0.70
SCORE_PNL_WEIGHT             = 0.30
