import re
from dataclasses import dataclass
from typing import Optional

@dataclass
class ParsedCommand:
    action: str  # "BUY", "SELL", "SHORT", "FLATTEN", "CANCEL"
    symbol: Optional[str] = None
    quantity: Optional[int] = None
    limit_price: Optional[float] = None
    raw_text: str = ""
    is_valid: bool = False
    error_msg: str = ""

class CommandParser:
    """
    Regex-based parser for natural language trading commands.
    Supported formats:
    - "Buy 100 AAPL"
    - "Sell 50 TSLA"
    - "Short 20 NVDA at 150.50"
    - "Flatten all"
    - "Cancel all"
    """
    
    def __init__(self):
        # Regex patterns
        self.buy_sell_pattern = re.compile(
            r'^(?P<action>buy|sell|short)\s+(?P<qty>\d+)\s+(?P<symbol>[A-Za-z]+)(?:\s+(?:at|@)\s+(?P<price>\d+(?:\.\d+)?))?', 
            re.IGNORECASE
        )
        self.flatten_pattern = re.compile(r'^(?:flatten|close)\s+(?:all|everything|positions)', re.IGNORECASE)
        self.cancel_pattern = re.compile(r'^cancel\s+(?:all|everything|orders)', re.IGNORECASE)
        
    def parse(self, text: str) -> ParsedCommand:
        text = text.strip()
        
        # 1. Flatten all
        if self.flatten_pattern.search(text):
            return ParsedCommand(action="FLATTEN", raw_text=text, is_valid=True)
            
        # 2. Cancel all
        if self.cancel_pattern.search(text):
            return ParsedCommand(action="CANCEL", raw_text=text, is_valid=True)
            
        # 3. Buy / Sell / Short
        match = self.buy_sell_pattern.search(text)
        if match:
            action = match.group('action').upper()

            qty_str = match.group('qty')
            symbol = match.group('symbol').upper()
            price_str = match.group('price')
            
            qty = int(qty_str) if qty_str else 0
            limit_price = float(price_str) if price_str else None
            
            if qty > 0 and symbol:
                return ParsedCommand(
                    action=action, 
                    symbol=symbol, 
                    quantity=qty, 
                    limit_price=limit_price, 
                    raw_text=text, 
                    is_valid=True
                )
            else:
                return ParsedCommand(
                    action=action, 
                    raw_text=text, 
                    is_valid=False, 
                    error_msg="Invalid quantity or symbol."
                )
                
        return ParsedCommand(
            action="UNKNOWN", 
            raw_text=text, 
            is_valid=False, 
            error_msg="Could not parse command. Use format: 'Buy 100 AAPL' or 'Flatten all'."
        )
