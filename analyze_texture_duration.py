# analyze_texture_duration.py
"""
TRANSITIONING texture ၏ ပျမ်းမျှ ကြာချိန်နှင့်
NEUTRAL/TRENDING သို့ ပြောင်းလဲသည့် Pattern ကို တွက်ချက်သော Script
"""

import os
import glob
import re
from datetime import datetime

LOG_FILES = sorted(glob.glob("logs/stock_engine.log*") + glob.glob("logs/daytrade_current.log"))

TEXTURE_PATTERN = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*texture=(\w+)"
)

def analyze_texture_duration():
    events = []

    for log_path in LOG_FILES:
        if not os.path.exists(log_path):
            continue
        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    m = TEXTURE_PATTERN.search(line)
                    if m:
                        try:
                            ts = datetime.fromisoformat(m.group(1))
                            texture = m.group(2)
                            if not events or events[-1][1] != texture:
                                events.append((ts, texture))
                        except Exception:
                            pass
        except Exception as e:
            pass

    if not events:
        print("No texture events found.")
        return

    # Sort chronological
    events.sort(key=lambda x: x[0])

    episodes = []
    i = 0
    while i < len(events):
        ts, tex = events[i]
        if tex == "TRANSITIONING":
            start_ts = ts
            j = i + 1
            while j < len(events) and events[j][1] == "TRANSITIONING":
                j += 1
            end_ts = events[j][0] if j < len(events) else None
            next_texture = events[j][1] if j < len(events) else "SESSION_END"
            if end_ts:
                duration_min = (end_ts - start_ts).total_seconds() / 60
                if 0 < duration_min < 1440:  # within a day
                    episodes.append({
                        "start": start_ts,
                        "end": end_ts,
                        "duration_min": round(duration_min, 1),
                        "next_texture": next_texture,
                    })
            i = j
        else:
            i += 1

    print(f"\n{'='*65}")
    print(f"  TEXTURE DURATION REPORT -- TRANSITIONING EPISODES")
    print(f"{'='*65}")
    print(f"  {'Start':<20} {'End':<20} {'Duration':>10} {'Next Texture':<15}")
    print(f"  {'-'*65}")
    for ep in episodes[-25:]:  # show recent 25
        print(f"  {ep['start'].strftime('%m-%d %H:%M:%S'):<20} "
              f"{ep['end'].strftime('%m-%d %H:%M:%S'):<20} "
              f"{ep['duration_min']:>8.1f}m  "
              f"{ep['next_texture']:<15}")

    if episodes:
        durations = [ep["duration_min"] for ep in episodes]
        avg_dur = sum(durations) / len(durations)
        max_dur = max(durations)
        min_dur = min(durations)
        next_counts = {}
        for ep in episodes:
            next_counts[ep["next_texture"]] = next_counts.get(ep["next_texture"], 0) + 1

        print(f"\n  {'-'*45}")
        print(f"  Total Transition Episodes : {len(episodes)}")
        print(f"  Avg Duration              : {avg_dur:.1f} min")
        print(f"  Min / Max Duration        : {min_dur:.1f} / {max_dur:.1f} min")
        print(f"  Next Texture after TRANSITIONING:")
        for tex, count in sorted(next_counts.items(), key=lambda x: -x[1]):
            print(f"    -> {tex:<20} {count:>3} times ({count/len(episodes)*100:.1f}%)")
    print(f"{'='*65}\n")

if __name__ == "__main__":
    analyze_texture_duration()
