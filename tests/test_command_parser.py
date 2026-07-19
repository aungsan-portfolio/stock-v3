import pytest
from strategies.command_parser import CommandParser

def test_command_parser():
    parser = CommandParser()
    
    # Buy commands
    cmd = parser.parse("Buy 100 AAPL")
    assert cmd.is_valid
    assert cmd.action == "BUY"
    assert cmd.quantity == 100
    assert cmd.symbol == "AAPL"
    
    cmd = parser.parse("buy 50 TSLA @ 150.25")
    assert cmd.is_valid
    assert cmd.action == "BUY"
    assert cmd.quantity == 50
    assert cmd.symbol == "TSLA"
    assert cmd.limit_price == 150.25
    
    # Sell / Short commands
    cmd = parser.parse("sell 10 NVDA")
    assert cmd.is_valid
    assert cmd.action == "SELL"
    assert cmd.quantity == 10
    
    cmd = parser.parse("short 5 AMD at 120")
    assert cmd.is_valid
    assert cmd.action == "SHORT"
    assert cmd.quantity == 5
    assert cmd.limit_price == 120.0
    
    # Flatten commands
    cmd = parser.parse("flatten all")
    assert cmd.is_valid
    assert cmd.action == "FLATTEN"
    
    cmd = parser.parse("close everything")
    assert cmd.is_valid
    assert cmd.action == "FLATTEN"
    
    # Cancel commands
    cmd = parser.parse("cancel all")
    assert cmd.is_valid
    assert cmd.action == "CANCEL"
    
    # Invalid commands
    cmd = parser.parse("buy AAPL")
    assert not cmd.is_valid
    
    cmd = parser.parse("hello world")
    assert not cmd.is_valid

if __name__ == "__main__":
    pytest.main(["-v", __file__])
