import xml.etree.ElementTree as ET
import urllib.request
import logging
import json
import os
import time
from typing import Dict, List, Optional

import config

logger = logging.getLogger(__name__)

CACHE_DIR = getattr(config, 'BASE_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / ".cache_data"
os.makedirs(CACHE_DIR, exist_ok=True)


# Keywords for sentiment
POSITIVE_WORDS = {"surge", "upgrade", "beat", "record", "launch", "approval", "higher", "buy", "growth", "jump"}
NEGATIVE_WORDS = {"downgrade", "lawsuit", "miss", "investigation", "recall", "lower", "sell", "decline", "drop", "plunge", "fall"}

# Keywords for catalysts
CATALYSTS = {
    "earnings": {"earnings", "revenue", "profit", "quarterly", "q1", "q2", "q3", "q4", "eps"},
    "product": {"launch", "announce", "unveil", "release", "event"},
    "analyst": {"upgrade", "downgrade", "target", "neutral", "rating", "analyst"},
    "legal": {"lawsuit", "investigation", "settle", "fine", "court", "probe"}
}

def analyze_news_sentiment(symbol: str) -> dict:
    """
    Fetches Yahoo Finance RSS for the given symbol, parses headlines,
    calculates sentiment score, and detects catalyst.
    """
    cache_file = os.path.join(CACHE_DIR, f"{symbol}_sentiment.json")
    ttl = getattr(config, 'SCAN_SENTIMENT_CACHE_TTL', 3600)
    
    if os.path.exists(cache_file):
        if time.time() - os.path.getmtime(cache_file) < ttl:
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.debug(f"Failed to read cache for {symbol}: {e}")

    urls_to_try = [
        f"http://finance.yahoo.com/rss/headline?s={symbol}",
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
    ]
    
    result = {
        "symbol": symbol,
        "headline_count": 0,
        "positive_count": 0,
        "negative_count": 0,
        "neutral_count": 0,
        "sentiment_score": 0.0,
        "recent_headlines": [],
        "catalyst_type": None,
        "top_headline": ""
    }
    
    xml_data = None
    items = []
    
    for url in urls_to_try:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as response:
                xml_data = response.read()
                
            root = ET.fromstring(xml_data)
            items = root.findall('./channel/item')
            if items:
                break
        except Exception as e:
            logger.debug(f"Failed fetching or parsing {url} for {symbol}: {e}")
            continue
            
    if not items:
        logger.warning(f"Failed to fetch valid news items for {symbol} from all endpoints.")
        return result
        
    try:
        root = ET.fromstring(xml_data)
        items = root.findall('./channel/item')
        
        if not items:
            return result
            
        items = items[:5]
        
        for item in items:
            title = item.find('title').text
            if not title: continue
            
            result["recent_headlines"].append(title)
            
            if not result["top_headline"]:
                result["top_headline"] = title
                
            title_lower = title.lower()
            
            words = set(title_lower.split())
            pos_matches = words.intersection(POSITIVE_WORDS)
            neg_matches = words.intersection(NEGATIVE_WORDS)
            
            if len(pos_matches) > len(neg_matches):
                result["positive_count"] += 1
            elif len(neg_matches) > len(pos_matches):
                result["negative_count"] += 1
            else:
                result["neutral_count"] += 1
                
            if not result["catalyst_type"]:
                for cat, keywords in CATALYSTS.items():
                    if words.intersection(keywords):
                        result["catalyst_type"] = cat
                        break
                        
        result["headline_count"] = len(result["recent_headlines"])
        
        if result["headline_count"] > 0:
            result["sentiment_score"] = (result["positive_count"] - result["negative_count"]) / result["headline_count"]
            
            try:
                with open(cache_file, 'w') as f:
                    json.dump(result, f)
            except Exception as e:
                logger.debug(f"Failed to write cache for {symbol}: {e}")
            
        return result
        
    except Exception as e:
        logger.warning(f"Failed to fetch news for {symbol}: {e}")
        return result
