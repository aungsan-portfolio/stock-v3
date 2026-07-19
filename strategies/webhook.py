import json
import logging
import urllib.request
import config

logger = logging.getLogger(__name__)

def send_discord_alert(message: str):
    url = getattr(config, "DISCORD_WEBHOOK_URL", "")
    if not url:
        return
    
    data = {"content": message}
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "DayTradingEngine/1.0"
        },
        method="POST"
    )
    
    try:
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        logger.warning(f"Failed to send Discord alert: {e}")
