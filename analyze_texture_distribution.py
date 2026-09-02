# analyze_texture_distribution.py
"""
stock_engine.log မှ Market Texture (TRANSITIONING / TRENDING / SIDEWAYS)
ဖြစ်ပေါ်မှု ရာခိုင်နှုန်းနှင့် ဖြစ်ပွားချိန်များကို ခွဲခြမ်းစိတ်ဖြာပေးသော Script
"""

import os
import glob
import re
from collections import defaultdict

LOG_DIR = "logs"
LOG_FILES = sorted(glob.glob("logs/stock_engine.log*") + glob.glob("logs/daytrade_current.log"))

# Regex: [REGIME] mode=CAUTION texture=TRANSITIONING ...
TEXTURE_PATTERN = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*texture=(\w+)"
)

def analyze_texture_distribution():
    daily_textures = defaultdict(list)

    for log_path in LOG_FILES:
        if not os.path.exists(log_path):
            continue
        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    match = TEXTURE_PATTERN.search(line)
                    if match:
                        ts_str, texture = match.group(1), match.group(2)
                        date_str = ts_str[:10]  # YYYY-MM-DD
                        daily_textures[date_str].append(texture)
        except Exception as e:
            print(f"Warning reading {log_path}: {e}")

    if not daily_textures:
        print("No texture logs found.")
        return

    print(f"\n{'='*65}")
    print(f"  TEXTURE DISTRIBUTION REPORT")
    print(f"{'='*65}")
    print(f"{'Date':<14} {'TRANSITIONING':>15} {'TRENDING':>10} {'SIDEWAYS':>10} {'Total':>7}")
    print(f"{'-'*65}")

    global_counts = defaultdict(int)

    for date in sorted(daily_textures.keys()):
        textures = daily_textures[date]
        total = len(textures)
        counts = defaultdict(int)
        for t in textures:
            counts[t] += 1
            global_counts[t] += 1

        trans_pct = counts["TRANSITIONING"] / total * 100 if total else 0
        trend_pct = counts["TRENDING"] / total * 100 if total else 0
        side_pct  = counts["SIDEWAYS"] / total * 100 if total else 0

        print(
            f"{date:<14} "
            f"{counts['TRANSITIONING']:>6} ({trans_pct:>5.1f}%)  "
            f"{counts['TRENDING']:>4} ({trend_pct:>4.1f}%)  "
            f"{counts['SIDEWAYS']:>4} ({side_pct:>4.1f}%)  "
            f"{total:>5}"
        )

    grand_total = sum(global_counts.values())
    print(f"{'-'*65}")
    if grand_total > 0:
        print(f"{'TOTAL':<14} "
              f"{global_counts['TRANSITIONING']:>6} "
              f"({global_counts['TRANSITIONING']/grand_total*100:>5.1f}%)  "
              f"{global_counts['TRENDING']:>4} "
              f"({global_counts['TRENDING']/grand_total*100:>4.1f}%)  "
              f"{global_counts['SIDEWAYS']:>4} "
              f"({global_counts['SIDEWAYS']/grand_total*100:>4.1f}%)  "
              f"{grand_total:>5}")
    print(f"{'='*65}\n")

if __name__ == "__main__":
    analyze_texture_distribution()
