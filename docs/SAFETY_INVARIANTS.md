# High-Reliability Trading Engine — Safety Invariants Contract

**Document Version**: 1.0.0  
**Effective Date**: 2026-09-05  
**Target Environment**: Day Trading Production Engine (`feature/p1.6-alpha-shadow`)  
**Purpose**: Formalize non-negotiable system-level invariants, explicit state transitions, and fail-closed policies to permanently eliminate accidental complexity and ad-hoc patch fatigue.

---

## 1. System Philosophy: Invariants-First Architecture

A high-reliability financial execution engine cannot rely on reactive local patches ("Bug detected → Add boolean guard → Pass test → Edge case appears → Add another guard"). 

Instead, every component must operate under strict, verifiable **Invariants**. Any operation that would breach an invariant MUST fail immediately and fail closed (rejecting trade execution rather than guessing or continuing in an unknown state).

---

## 2. Order Idempotency & Lifecycle Invariants (`order_idempotency.py`)

| Invariant ID | Name | Formal Statement | Failure Action |
|---|---|---|---|
| **I-1** | **Durable Reservation** | An order MUST NOT be submitted to the broker unless its reservation record has been durably persisted to non-volatile disk storage. | **Fail-Closed**: If `_save_state()` fails (disk I/O error, permissions, full volume), `reserve()` returns `(False, reason)` and order execution halts. Memory-only reservations are forbidden. |
| **I-2** | **Broker Ambiguity Hold** | If the broker's receipt of an order is unconfirmed, ambiguous, or timed out during submission, the reservation MUST NOT be released. | **Reconciliation Required**: State transitions to `RECONCILIATION_REQUIRED`. The slot remains locked until active broker polling or manual operator reconciliation confirms the order status. |
| **I-3** | **Transport Retries Reuse Client ID** | Repeated transport submissions of the exact same signal intent (due to network drops, HTTP 504/502, or connection timeouts) MUST reuse the identical `client_order_id`. | `submission_retry_attempt` is strictly excluded from the deterministic digest calculation, guaranteeing broker-side deduplication. |
| **I-4** | **Strategy Re-entry Uniqueness** | Deliberate strategy-level re-entries, scale-ins, or post-exit re-engagements MUST produce a distinct, unique `client_order_id`. | `strategy_entry_attempt` is incremented and included in the deterministic digest calculation, guaranteeing a fresh broker ID. |
| **I-5** | **Terminal-Only Acknowledged Release** | An `ACKNOWLEDGED` reservation (where a live parent order exists at the broker) CANNOT be released by standard `release()`. | **Prohibited Transition**: `release()` strictly checks status. Calling `release()` on an `ACKNOWLEDGED` order raises an error or logs an invariant breach and refuses deletion. It can ONLY transition via definitive broker terminal events (`filled`, `canceled`, `rejected`, `expired`). |
| **I-6** | **Corrupt State Fails Closed** | If the persistent idempotency state file is missing, unreadable, or corrupted on engine boot, the engine MUST NOT assume an empty state. | **Fail-Closed Engine Health**: The store marks itself `is_healthy = False`. All subsequent `reserve()` requests are blocked until the state file is repaired or formally reset. |

---

## 3. Order Lifecycle State Machine

```
              ┌───────────────┐
              │      NEW      │
              └───────┬───────┘
                      │ reserve() [must durably persist]
                      ▼
              ┌───────────────┐
              │   RESERVED    │
              └───────┬───────┘
                      │ submit_order() initiated
                      ▼
              ┌───────────────┐
        ┌─────│  SUBMITTING   │─────┐
        │     └───────────────┘     │
        │ broker               broker │ timeout /
        │ ack                  reject │ unknown
        ▼                             ▼
┌───────────────┐             ┌─────────────────────────┐
│ ACKNOWLEDGED  │             │ RECONCILIATION_REQUIRED │
└───────┬───────┘             └─────────────┬───────────┘
        │                                   │ broker lookup confirms:
        │ terminal broker                   ├─ order found -> ACKNOWLEDGED
        │ event                             └─ order absent -> RELEASED
        ▼                                           │
┌───────────────┐                     ┌─────────────▼───────────┐
│   TERMINAL    │                     │        RELEASED         │
│ (filled/canc) │                     │     (slot available)    │
└───────────────┘                     └─────────────────────────┘
```

### Transition Validation Rules

1. `NEW → RESERVED`: Permitted only if atomic disk write succeeds.
2. `RESERVED → SUBMITTING`: Order submission payload prepared.
3. `SUBMITTING → ACKNOWLEDGED`: Broker returns valid parent order ID (`status in {'new', 'accepted', 'pending_new'}`).
4. `SUBMITTING → RELEASED`: Broker definitively rejects submission *before* order ingestion (e.g., 400 Bad Request, pre-submit account equity veto).
5. `SUBMITTING → RECONCILIATION_REQUIRED`: Broker connection drops, 422 client_order_id collision occurs, or request times out without a definitive HTTP response.
6. `ACKNOWLEDGED → TERMINAL`: Allowed only upon broker websocket / polling receipt of terminal status (`filled`, `canceled`, `rejected`, `expired`).
7. **DISALLOWED TRANSITION**: `ACKNOWLEDGED → RELEASED`. (Attempting to wipe a live broker position via routine cleanup is an invariant breach).

---

## 4. Signal & Strategy Geometry Invariants (`VWAPBounceStrategy`)

| Invariant ID | Name | Formal Statement | Enforcement Mechanism |
|---|---|---|---|
| **G-1** | **Long Geometry Positivity & Ordering** | Long setup prices must strictly satisfy: $$0 < \text{stop\_price} < \text{entry\_price} < \text{target\_price}$$ | Pre-signal emission assertion. Rejects setup immediately if broken. |
| **G-2** | **Short Geometry Positivity & Ordering** | Short setup prices must strictly satisfy: $$0 < \text{target\_price} < \text{entry\_price} < \text{stop\_price}$$ | Pre-signal emission assertion. Rejects setup immediately if broken. |
| **G-3** | **Executable Risk Consistency** | The signal's declared `risk_per_share` MUST strictly equal the post-rounding executable stop distance: $$\text{risk\_per\_share} \equiv \lvert \text{entry\_price} - \text{stop\_price} \rvert$$ | Executable stop price is rounded first; `risk_per_share` is computed directly from `round(abs(entry - stop), precision)`. Pre-rounding risk is never emitted. |
| **G-4** | **Direction-Aware Noise Floor Preservation** | Rounding to the tick grid must never shrink actual risk below the noise floor: $$\text{actual\_risk} \ge \max(0.75 \times \text{ATR},\; 1.5 \times \text{spread},\; \text{tick\_size})$$ | Long stops floor down (`_floor_to_tick`); short stops ceil up (`_ceil_to_tick`). Pushes stop away from entry price, guaranteeing zero noise-floor erosion. |
| **G-5** | **Structure-First Target Validation** | Targets MUST NOT be artificially force-extended beyond structural boundaries: $$\text{actual\_rr} \ge \text{MIN\_RR}$$ | Structural reward distance is clamped to resistance/support. If resulting R:R $< 1.0\text{R}$, signal is rejected cleanly upstream. Target inflation (`max(target, risk * min_rr)`) is prohibited. |
| **G-6** | **Structural Integrity (No Inversion Masking)** | Structural stop MUST be strictly on the defensive side of market price: Long: $\text{price} > \text{structural\_stop}$ Short: $\text{structural\_stop} > \text{price}$ | If $\text{raw\_risk} \le 0$ or non-finite, setup is fundamentally broken. Rejected immediately; never masked by noise-floor substitution. |

---

## 5. Failure Policy: Fail-Closed vs Fail-Open Rules

In mission-critical automated execution, the cardinal rule is:

> **When uncertainty arises, fail closed. It is vastly better to miss ten valid trades than to execute a single uncontrolled, unmanaged, or duplicate order.**

1. **State Persistence**: Disk save error = **Fail Closed** (Refuse order).
2. **State Load**: Corrupted JSON = **Fail Closed** (Halt order submissions until inspected).
3. **Execution Quote**: Stale quote ($> 10\text{s}$) or wide spread ($> 0.75\%$) = **Fail Closed** (Veto submission).
4. **Broker Timeout**: HTTP timeout during submit = **Fail Closed** (Lock symbol into `RECONCILIATION_REQUIRED`, do NOT release slot).
5. **Structural Risk**: Broken candle geometry = **Fail Closed** (Reject signal).

---

## 6. Verification & Audit Sign-Off

This safety contract is audited via targeted, invariant-named test suites:
- `test_persistence_failure_blocks_submission`
- `test_corrupt_state_fails_closed`
- `test_acknowledged_reservation_cannot_be_manually_released`
- `test_timeout_after_broker_acceptance_does_not_release`
- `test_transport_retry_reuses_client_order_id`
- `test_reentry_changes_client_order_id`
- `test_vwap_geometry_numerical_safety_invariants`
