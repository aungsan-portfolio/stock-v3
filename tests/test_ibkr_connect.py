"""Manual paper-connect smoke check (NOT a unit test).

Run it directly to verify a paper TWS/Gateway is reachable on port 7497:

    python test_ibkr_connect.py

The connect attempt is guarded under ``if __name__ == "__main__"`` so that
``python -m unittest discover -p "test_*.py"`` can IMPORT this module without
opening a real socket. (Phase 5B-2 wraps connect() in a bounded backoff retry,
so an unguarded import-time connect would sleep through the backoff schedule
during discovery.)
"""
import asyncio
import sys

if sys.platform == "win32" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


if __name__ == "__main__":
    from ibkr_bridge import IBKRBridge

    b = IBKRBridge()
    print("connected=", b.connect())
    b.disconnect()
