import os
import json
import shutil
import hashlib
from datetime import datetime, timezone

def calculate_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def execute_aapl_quarantine():
    state_path = "data/trailing_state.json"
    assert os.path.exists(state_path), f"{state_path} does not exist"

    # 1. Compute pre-quarantine hash
    pre_hash = calculate_sha256(state_path)
    now_utc = datetime.now(timezone.utc).isoformat()
    timestamp_str = "20261013_221230"

    # 2. Unique timestamped backup
    backup_path = f"data/trailing_state.json.bak_pre_quarantine_{timestamp_str}"
    shutil.copy2(state_path, backup_path)
    assert os.path.exists(backup_path)
    assert calculate_sha256(backup_path) == pre_hash

    # 3. Read and verify state contents
    with open(state_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pending_exits = data.get("__pending_exits__", {})
    assert len(pending_exits) == 8, f"Expected 8 pending exits, found {len(pending_exits)}"

    target_key = "AAPL:val_test_order_123"
    assert target_key in pending_exits, f"Target key {target_key} not found in pending exits"

    target_record = dict(pending_exits[target_key])

    # Precondition assertion on record values
    assert target_record.get("symbol") == "AAPL"
    assert target_record.get("qty") == 100.0
    assert target_record.get("entry_price") == 150.0
    assert target_record.get("stop_price") == 140.0
    assert target_record.get("stop_order_id") == "val_test_order_123"
    assert target_record.get("strategy") == "MOMENTUM_SCALP"

    # 4. Save Durable Quarantine Artifact FIRST
    artifact_path = "data/quarantine_AAPL_val_test_order_123_artifact.json"
    artifact = {
        "artifact_version": "1.0",
        "action": "CONTROLLED_QUARANTINE",
        "timestamp_utc": now_utc,
        "source_state_file": os.path.abspath(state_path),
        "source_state_sha256_pre": pre_hash,
        "backup_path": os.path.abspath(backup_path),
        "quarantined_key": target_key,
        "quarantined_record": target_record,
        "classification": "CONFIRMED_TEST_FIXTURE_CONTAMINATION",
        "fixture_provenance": {
            "source_test": "tests/test_journal_isolation.py:112",
            "test_function": "test_emergency_flatten_validation_routing",
            "exact_matches": [
                "symbol == 'AAPL'",
                "entry_price == 150.0",
                "stop_price == 140.0",
                "qty == 100.0",
                "order_id == 'val_test_order_123'",
                "strategy == 'MOMENTUM_SCALP'"
            ]
        },
        "root_cause": "Prior to state-isolation remediation, DynamicTrailingStopManager wrote test fixtures into global module STATE_FILE data/trailing_state.json",
        "operational_impact": "Alpaca API rejected background trailing-stop reconciliation queries with 'badly formed hexadecimal UUID string' due to mock non-UUID order ID",
        "retained_episodes_count": 7,
        "retained_symbols": ["UBER (7 episodes)", "Top-level UBER state"]
    }

    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

    assert os.path.exists(artifact_path)
    print(f"Quarantine artifact durably written to {artifact_path}")

    # 5. Remove ONLY the target key from pending exits
    del pending_exits[target_key]
    assert len(pending_exits) == 7, f"Expected 7 pending exits after removal, found {len(pending_exits)}"
    assert target_key not in pending_exits

    # Ensure all 7 UBER records remain completely intact
    uber_keys = [k for k in pending_exits.keys() if k.startswith("UBER:")]
    assert len(uber_keys) == 7, f"Expected 7 UBER pending keys retained, found {len(uber_keys)}"

    # Ensure top-level UBER state remains completely intact
    assert "UBER" in data, "Top level UBER state was inadvertently modified"
    assert data["UBER"].get("last_stop_order_id") == "val_stop_uber_99"

    # 6. Atomic replacement of state file
    tmp_path = f"{state_path}.tmp_quarantine_{timestamp_str}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_path, state_path)

    # 7. Post-quarantine verification
    post_hash = calculate_sha256(state_path)
    with open(state_path, "r", encoding="utf-8") as f:
        post_data = json.load(f)

    post_pending = post_data.get("__pending_exits__", {})
    assert len(post_pending) == 7, f"Post-quarantine count mismatch: {len(post_pending)}"
    assert target_key not in post_pending
    assert "UBER" in post_data
    assert len([k for k in post_pending.keys() if k.startswith("UBER:")]) == 7

    print("==================================================")
    print("CONTROLLED QUARANTINE SUCCESSFUL")
    print(f"Pre-quarantine SHA256:  {pre_hash}")
    print(f"Post-quarantine SHA256: {post_hash}")
    print(f"Pending exits count: 8 -> 7")
    print(f"Quarantined key: {target_key}")
    print(f"Retained UBER pending keys: {len(uber_keys)}")
    print(f"Top-level UBER state: PRESERVED")
    print("==================================================")

if __name__ == "__main__":
    execute_aapl_quarantine()
