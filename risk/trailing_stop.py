"""
Trailing Stop Loss Module

This module implements a trailing stop mechanism that locks in profits
by adjusting the stop loss level as the price moves favorably. The stop
"trails" behind the price, protecting gains while allowing continued upside.

Key Concepts:
    - Trailing Stop: Dynamic stop loss that moves with favorable price action
    - Trail Percentage: How far behind the peak the stop should trail
    - Peak Price: Highest price achieved (for long) or lowest (for short)
    - Position Average: Entry price of the position

Usage:
    stop = TrailingStop(side='long')
    stop.update(current_price=105, position_avg_price=100)
    if stop.should_exit(trail_pct=0.015):
        print("Exit signal triggered!")
"""

import logging
from typing import Optional, Dict, Literal
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrailingStop:
    """
    Dynamic trailing stop loss manager for position protection.
    
    This class tracks price movements and adjusts stop loss levels
    dynamically to lock in profits while allowing continued gains.
    Works for both long and short positions.
    
    Attributes:
        side (str): Position side - 'long' or 'short'
        peak_price (float): Best price achieved since initialization
        current_price (float): Latest price update
        position_avg_price (float): Entry price of position
        stop_level (float): Current stop loss trigger level
        trail_pct (float): Trailing percentage (e.g., 0.015 for 1.5%)
        is_active (bool): Whether trailing stop is active
        trigger_time (datetime): When exit was triggered
    """
    
    def __init__(self, side: Literal['long', 'short'] = 'long'):
        """
        Initialize trailing stop manager.
        
        Args:
            side: Position side - 'long' or 'short' (default: 'long')
            
        Raises:
            ValueError: If side is invalid
            
        Example:
            >>> stop = TrailingStop(side='long')
            >>> stop.update(current_price=100, position_avg_price=98)
        """
        if side not in ['long', 'short']:
            raise ValueError(f"side must be 'long' or 'short', got '{side}'")
        
        self.side = side
        self.peak_price: Optional[float] = None
        self.current_price: Optional[float] = None
        self.position_avg_price: Optional[float] = None
        self.stop_level: Optional[float] = None
        self.trail_pct: float = 0.0
        self.is_active = False
        self.triggered = False
        self.trigger_time: Optional[datetime] = None
        
        # Statistics
        self.update_count = 0
        self.max_favorable_move_pct = 0.0
        
        logger.info(f"TrailingStop initialized for {side} position")
    
    def update(self, current_price: float, position_avg_price: float) -> None:
        """
        Update trailing stop with current market price.
        
        This method should be called regularly (e.g., on each price tick or candle)
        to update the trailing stop calculation.
        
        Args:
            current_price: Current market price
            position_avg_price: Average entry price of position
            
        Raises:
            ValueError: If prices are invalid
            
        Example:
            >>> stop = TrailingStop(side='long')
            >>> stop.update(current_price=102, position_avg_price=100)
            >>> print(f"Peak price: ${stop.peak_price:.2f}")
        """
        if current_price <= 0:
            raise ValueError(f"current_price must be positive, got {current_price}")
        
        if position_avg_price <= 0:
            raise ValueError(f"position_avg_price must be positive, got {position_avg_price}")
        
        self.current_price = current_price
        self.position_avg_price = position_avg_price
        self.update_count += 1
        
        # Initialize peak price on first update
        if self.peak_price is None:
            self.peak_price = current_price
            logger.debug(f"Initial peak price set: ${self.peak_price:.2f}")
        
        # Update peak price based on position side
        if self.side == 'long':
            # For long positions, track highest price
            if current_price > self.peak_price:
                self.peak_price = current_price
                logger.debug(f"New peak (long): ${self.peak_price:.2f}")
        else:
            # For short positions, track lowest price
            if current_price < self.peak_price:
                self.peak_price = current_price
                logger.debug(f"New peak (short): ${self.peak_price:.2f}")
        
        # Calculate max favorable move percentage
        if self.side == 'long':
            favorable_move_pct = (self.peak_price - self.position_avg_price) / self.position_avg_price
        else:
            favorable_move_pct = (self.position_avg_price - self.peak_price) / self.position_avg_price
        
        self.max_favorable_move_pct = max(self.max_favorable_move_pct, favorable_move_pct)
    
    def should_exit(self, trail_pct: float = 0.015) -> bool:
        """
        Check if trailing stop exit condition is met.
        
        This is the main decision method that determines if the position
        should be exited based on the trailing stop logic.
        
        Args:
            trail_pct: Trailing percentage as decimal (default: 0.015 for 1.5%)
            
        Returns:
            bool: True if exit signal triggered, False otherwise
            
        Raises:
            ValueError: If trail_pct is invalid or update() not called
            
        Example:
            >>> stop = TrailingStop(side='long')
            >>> stop.update(current_price=105, position_avg_price=100)
            >>> if stop.should_exit(trail_pct=0.015):
            ...     print("Exit long position - stop triggered!")
        """
        if not 0 < trail_pct < 1:
            raise ValueError(f"trail_pct must be between 0 and 1, got {trail_pct}")
        
        if self.current_price is None or self.peak_price is None:
            raise ValueError("Must call update() before should_exit()")
        
        self.trail_pct = trail_pct
        
        # Calculate stop level based on position side
        if self.side == 'long':
            # For long: stop is below peak by trail_pct
            self.stop_level = self.peak_price * (1 - trail_pct)
            exit_triggered = self.current_price <= self.stop_level
        else:
            # For short: stop is above peak by trail_pct
            self.stop_level = self.peak_price * (1 + trail_pct)
            exit_triggered = self.current_price >= self.stop_level
        
        # Handle trigger event
        if exit_triggered and not self.triggered:
            self.triggered = True
            self.trigger_time = datetime.now()
            
            # Calculate exit statistics
            if self.side == 'long':
                exit_pnl_pct = (self.current_price - self.position_avg_price) / self.position_avg_price * 100
            else:
                exit_pnl_pct = (self.position_avg_price - self.current_price) / self.position_avg_price * 100
            
            logger.warning(
                f"🛑 TRAILING STOP TRIGGERED ({self.side.upper()})\n"
                f"  Current Price: ${self.current_price:.2f}\n"
                f"  Stop Level: ${self.stop_level:.2f}\n"
                f"  Peak Price: ${self.peak_price:.2f}\n"
                f"  Entry Price: ${self.position_avg_price:.2f}\n"
                f"  Exit P&L: {exit_pnl_pct:+.2f}%\n"
                f"  Max Move: {self.max_favorable_move_pct*100:+.2f}%"
            )
        elif not exit_triggered:
            # Normal operation - log status
            distance_from_stop = abs(self.current_price - self.stop_level) / self.current_price * 100
            logger.debug(
                f"Trailing stop active: price=${self.current_price:.2f}, "
                f"stop=${self.stop_level:.2f}, distance={distance_from_stop:.2f}%"
            )
        
        return exit_triggered
    
    def get_status(self) -> Dict[str, any]:
        """
        Get current trailing stop status and statistics.
        
        Returns:
            Dict with comprehensive status information
            
        Example:
            >>> status = stop.get_status()
            >>> print(f"Stop level: ${status['stop_level']:.2f}")
        """
        if self.current_price and self.stop_level:
            if self.side == 'long':
                distance_to_stop = self.current_price - self.stop_level
            else:
                distance_to_stop = self.stop_level - self.current_price
            
            distance_to_stop_pct = (distance_to_stop / self.current_price) * 100
        else:
            distance_to_stop = None
            distance_to_stop_pct = None
        
        # Calculate current P&L if we have all data
        current_pnl_pct = None
        if self.current_price and self.position_avg_price:
            if self.side == 'long':
                current_pnl_pct = (self.current_price - self.position_avg_price) / self.position_avg_price * 100
            else:
                current_pnl_pct = (self.position_avg_price - self.current_price) / self.position_avg_price * 100
        
        return {
            'side': self.side,
            'current_price': self.current_price,
            'position_avg_price': self.position_avg_price,
            'peak_price': self.peak_price,
            'stop_level': self.stop_level,
            'trail_pct': self.trail_pct * 100 if self.trail_pct else None,
            'triggered': self.triggered,
            'trigger_time': self.trigger_time,
            'distance_to_stop': distance_to_stop,
            'distance_to_stop_pct': distance_to_stop_pct,
            'current_pnl_pct': current_pnl_pct,
            'max_favorable_move_pct': self.max_favorable_move_pct * 100,
            'update_count': self.update_count
        }
    
    def reset(self):
        """
        Reset the trailing stop for a new position.
        
        Example:
            >>> stop.reset()  # Ready for next position
        """
        self.peak_price = None
        self.current_price = None
        self.position_avg_price = None
        self.stop_level = None
        self.trail_pct = 0.0
        self.triggered = False
        self.trigger_time = None
        self.update_count = 0
        self.max_favorable_move_pct = 0.0
        
        logger.info(f"TrailingStop reset for {self.side} position")


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage demonstrating trailing stop in different scenarios.
    """
    
    print("\n" + "=" * 70)
    print("TRAILING STOP - Example Usage")
    print("=" * 70)
    print()
    
    # Scenario 1: Long position with profitable move
    print("-" * 70)
    print("SCENARIO 1: LONG POSITION - Profitable Move")
    print("-" * 70)
    print()
    
    stop_long = TrailingStop(side='long')
    entry_price = 100.0
    trail_percentage = 0.015  # 1.5% trailing stop
    
    print(f"Position Setup:")
    print(f"  Side: LONG")
    print(f"  Entry Price: ${entry_price:.2f}")
    print(f"  Trail Percentage: {trail_percentage*100:.2f}%")
    print()
    
    # Simulate price movement
    price_movements = [
        (100, "Entry"),
        (102, "Small gain +2%"),
        (105, "Continuing up +5%"),
        (108, "Peak reached +8%"),
        (107, "Small pullback to +7%"),
        (106.5, "Further pullback"),
        (106.378, "At stop level - should trigger"),
    ]
    
    print("Price Movement:")
    print()
    
    for price, description in price_movements:
        print(f"  Price: ${price:.2f} - {description}")
        
        stop_long.update(current_price=price, position_avg_price=entry_price)
        
        should_exit = stop_long.should_exit(trail_pct=trail_percentage)
        status = stop_long.get_status()
        
        print(f"    Peak: ${status['peak_price']:.2f}")
        print(f"    Stop Level: ${status['stop_level']:.2f}")
        print(f"    Current P&L: {status['current_pnl_pct']:+.2f}%")
        
        if should_exit:
            print(f"    🛑 EXIT TRIGGERED!")
            break
        else:
            print(f"    ✓ Hold position (distance to stop: {status['distance_to_stop_pct']:.2f}%)")
        
        print()
    
    # Scenario 2: Short position
    print("-" * 70)
    print("SCENARIO 2: SHORT POSITION - Profitable Move")
    print("-" * 70)
    print()
    
    stop_short = TrailingStop(side='short')
    entry_price_short = 100.0
    
    print(f"Position Setup:")
    print(f"  Side: SHORT")
    print(f"  Entry Price: ${entry_price_short:.2f}")
    print(f"  Trail Percentage: {trail_percentage*100:.2f}%")
    print()
    
    # Simulate price movement for short
    price_movements_short = [
        (100, "Entry"),
        (98, "Profitable move -2%"),
        (95, "Continuing down -5%"),
        (92, "Peak profit -8%"),
        (93, "Small bounce +1% from low"),
        (93.38, "At stop level - should trigger"),
    ]
    
    print("Price Movement:")
    print()
    
    for price, description in price_movements_short:
        print(f"  Price: ${price:.2f} - {description}")
        
        stop_short.update(current_price=price, position_avg_price=entry_price_short)
        
        should_exit = stop_short.should_exit(trail_pct=trail_percentage)
        status = stop_short.get_status()
        
        print(f"    Peak: ${status['peak_price']:.2f}")
        print(f"    Stop Level: ${status['stop_level']:.2f}")
        print(f"    Current P&L: {status['current_pnl_pct']:+.2f}%")
        
        if should_exit:
            print(f"    🛑 EXIT TRIGGERED!")
            break
        else:
            print(f"    ✓ Hold position (distance to stop: {status['distance_to_stop_pct']:.2f}%)")
        
        print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Trailing Stop Mechanics:

LONG Position:
  • Stop trails BELOW the peak price by trail_pct
  • Exit when: current_price <= peak_price * (1 - trail_pct)
  • Protects profits by locking in gains as price rises
  • Example: Peak=$108, Trail=1.5% → Stop=$106.38

SHORT Position:
  • Stop trails ABOVE the peak (lowest) price by trail_pct
  • Exit when: current_price >= peak_price * (1 + trail_pct)
  • Protects profits by locking in gains as price falls
  • Example: Peak=$92, Trail=1.5% → Stop=$93.38

Benefits:
  ✓ Locks in profits automatically
  ✓ Allows continued upside/downside capture
  ✓ Removes emotion from exit decisions
  ✓ Adapts to market volatility
""")
    print("=" * 70)
    print()
