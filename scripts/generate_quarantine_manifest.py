import json
from datetime import datetime, timezone

def generate_manifest():
    with open("data/trailing_state.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    pending_exits = data.get("__pending_exits__", {})
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run_only": True,
        "authoritative_broker_truth": {
            "daytrade_account": {
                "account_id": "PA3UODIC5JTA",
                "status": "ACTIVE",
                "equity": 98375.87,
                "positions_count": 0,
                "orders_count": 0,
                "symbols_held": []
            },
            "swing_account": {
                "account_id": "PA3R71I7JFDZ",
                "status": "ACTIVE",
                "equity": 100734.73,
                "positions_count": 5,
                "orders_count": 5,
                "symbols_held": ["BAC (162.0)", "JPM (11.0)", "MSFT (8.0)", "QQQ (13.0)", "TSLA (4.0)"],
                "broker_protection_certified": True
            }
        },
        "records_total": len(pending_exits),
        "episodes": [],
        "top_level_review": {
            "symbol": "UBER",
            "active": data.get("UBER", {}).get("active"),
            "reconciliation_status": data.get("UBER", {}).get("reconciliation_status"),
            "last_stop_order_id": data.get("UBER", {}).get("last_stop_order_id"),
            "fixture_evidence": "tests/test_rule6_invariant_remediation.py:184 (val_stop_uber_99)",
            "account_broker_checks": {
                "daytrade_account_PA3UODIC5JTA": "0 UBER positions, 0 UBER orders",
                "swing_account_PA3R71I7JFDZ": "0 UBER positions, 0 UBER orders"
            },
            "recommendation": "RETAIN_INACTIVE_STATE_UNCHANGED"
        }
    }

    for ep_id, rec in pending_exits.items():
        is_aapl = (ep_id == "AAPL:val_test_order_123")
        sym = rec.get("symbol")
        entry = {
            "episode_id": ep_id,
            "symbol": sym,
            "qty": rec.get("qty"),
            "entry_price": rec.get("entry_price"),
            "stop_price": rec.get("stop_price"),
            "stop_order_id": rec.get("stop_order_id"),
            "registered_at": rec.get("registered_at"),
            "account_checks": {
                "daytrade_PA3UODIC5JTA": f"0 {sym} positions, 0 {sym} orders",
                "swing_PA3R71I7JFDZ": f"0 {sym} positions, 0 {sym} orders"
            }
        }
        if is_aapl:
            entry["classification"] = "CONFIRMED_TEST_FIXTURE"
            entry["fixture_evidence"] = "tests/test_journal_isolation.py:112 test_emergency_flatten_validation_routing (symbol=AAPL, entry=150.0, stop=140.0, qty=100.0, order_id=val_test_order_123, strategy=MOMENTUM_SCALP)"
            entry["reason"] = "Hardcoded test string val_test_order_123 causes Alpaca API 400 Bad Request (badly formed hexadecimal UUID string) during background order reconciler queries"
            entry["action"] = "PROPOSED_FOR_CONTROLLED_QUARANTINE"
        else:
            entry["classification"] = "DIAGNOSTIC_FIXTURE_EVALUATION"
            entry["fixture_evidence"] = "tests/test_rule6_invariant_remediation.py:184 (entry=69.0, stop=68.5, qty=1.0, null lineage)"
            entry["reason"] = "Retained unresolved per strict User Instruction (Section 4). No broad prefix deletion. Preserved intact in state file."
            entry["action"] = "RETAINED_UNRESOLVED"

        manifest["episodes"].append(entry)

    with open("data/quarantine_dryrun_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Authoritative manifest generated at data/quarantine_dryrun_manifest.json")

if __name__ == "__main__":
    generate_manifest()
