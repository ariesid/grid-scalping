"""
Drawdown Monitor and Protection Module

This module tracks portfolio drawdown and provides protection mechanisms
to prevent excessive losses. It monitors both realized and unrealized P&L
to calculate current drawdown from peak equity.

Key Concepts:
    - Drawdown: % decline from peak equity to current equity
    - Max Drawdown: Maximum allowed drawdown before triggering protection
    - Peak Equity: Highest portfolio value achieved
    - Current Equity: Current portfolio value (capital + P&L)

Usage:
    protector = DrawdownProtector(initial_capital=10000, max_dd=0.05)
    if protector.update(realized_pnl=100, unrealized_pnl=-300):
        print("Max drawdown breached! Stop trading!")
"""

import logging
from typing import Dict, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DrawdownProtector:
    """
    Monitor and protect against excessive portfolio drawdown.
    
    This class tracks portfolio equity over time, identifies peak values,
    and calculates current drawdown. When drawdown exceeds the specified
    threshold, it triggers protection mechanisms.
    
    Attributes:
        initial_capital (float): Starting capital
        max_dd (float): Maximum allowed drawdown as decimal (e.g., 0.05 for 5%)
        peak_equity (float): Highest equity achieved
        current_equity (float): Current portfolio equity
        current_drawdown (float): Current drawdown percentage
        breach_triggered (bool): Whether max drawdown has been breached
        breach_time (datetime): When breach occurred
    """
    
    def __init__(self, initial_capital: float, max_dd: float = 0.05):
        """
        Initialize drawdown protector.
        
        Args:
            initial_capital: Starting capital amount (must be positive)
            max_dd: Maximum allowed drawdown as decimal (default: 0.05 for 5%)
            
        Raises:
            ValueError: If initial_capital <= 0 or max_dd invalid
            
        Example:
            >>> protector = DrawdownProtector(initial_capital=10000, max_dd=0.05)
            >>> print(f"Max allowed loss: ${10000 * 0.05:.2f}")
        """
        if initial_capital <= 0:
            raise ValueError(f"initial_capital must be positive, got {initial_capital}")
        
        if not 0 < max_dd < 1:
            raise ValueError(f"max_dd must be between 0 and 1, got {max_dd}")
        
        self.initial_capital = initial_capital
        self.max_dd = max_dd
        self.peak_equity = initial_capital
        self.current_equity = initial_capital
        self.current_drawdown = 0.0
        self.breach_triggered = False
        self.breach_time: Optional[datetime] = None
        
        # Statistics tracking
        self.max_drawdown_seen = 0.0
        self.update_count = 0
        
        logger.info(
            f"DrawdownProtector initialized: capital=${initial_capital:.2f}, "
            f"max_dd={max_dd*100:.1f}% (${initial_capital * max_dd:.2f})"
        )
    
    def update(self, realized_pnl: float, unrealized_pnl: float) -> bool:
        """
        Update equity and check for drawdown breach.
        
        This is the main method that should be called regularly to update
        the portfolio state and check if protection should be triggered.
        
        Args:
            realized_pnl: Realized profit/loss (closed positions)
            unrealized_pnl: Unrealized profit/loss (open positions)
            
        Returns:
            bool: True if max drawdown breached, False otherwise
            
        Example:
            >>> protector = DrawdownProtector(10000, 0.05)
            >>> # After some trading...
            >>> if protector.update(realized_pnl=-200, unrealized_pnl=-300):
            ...     print("STOP TRADING - Max drawdown reached!")
        """
        # Calculate current equity
        total_pnl = realized_pnl + unrealized_pnl
        self.current_equity = self.initial_capital + total_pnl
        
        # Update peak equity if we've reached a new high
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity
            logger.debug(f"New peak equity reached: ${self.peak_equity:.2f}")
        
        # Calculate current drawdown
        if self.peak_equity > 0:
            self.current_drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
        else:
            self.current_drawdown = 0.0
        
        # Track max drawdown seen
        if self.current_drawdown > self.max_drawdown_seen:
            self.max_drawdown_seen = self.current_drawdown
        
        # Update statistics
        self.update_count += 1
        
        # Check for breach
        is_breach = self.current_drawdown >= self.max_dd
        
        if is_breach and not self.breach_triggered:
            # First time breach
            self.breach_triggered = True
            self.breach_time = datetime.now()
            
            logger.critical(
                f"⚠️  MAX DRAWDOWN BREACHED! ⚠️\n"
                f"  Current Drawdown: {self.current_drawdown*100:.2f}%\n"
                f"  Max Allowed: {self.max_dd*100:.2f}%\n"
                f"  Peak Equity: ${self.peak_equity:.2f}\n"
                f"  Current Equity: ${self.current_equity:.2f}\n"
                f"  Loss: ${self.peak_equity - self.current_equity:.2f}"
            )
        elif is_breach:
            # Continued breach
            logger.warning(
                f"Drawdown breach continues: {self.current_drawdown*100:.2f}% "
                f"(Max: {self.max_dd*100:.2f}%)"
            )
        else:
            # Normal operation
            logger.debug(
                f"Drawdown: {self.current_drawdown*100:.2f}% "
                f"(Safe: {self.max_dd*100:.2f}%), "
                f"Equity: ${self.current_equity:.2f}"
            )
        
        return is_breach
    
    def get_status(self) -> Dict[str, any]:
        """
        Get current drawdown protection status.
        
        Returns:
            Dict with comprehensive status information
            
        Example:
            >>> status = protector.get_status()
            >>> print(f"Current DD: {status['current_drawdown_pct']:.2f}%")
        """
        return {
            'initial_capital': self.initial_capital,
            'peak_equity': self.peak_equity,
            'current_equity': self.current_equity,
            'current_drawdown': self.current_drawdown,
            'current_drawdown_pct': self.current_drawdown * 100,
            'max_allowed_dd': self.max_dd,
            'max_allowed_dd_pct': self.max_dd * 100,
            'max_drawdown_seen': self.max_drawdown_seen,
            'max_drawdown_seen_pct': self.max_drawdown_seen * 100,
            'breach_triggered': self.breach_triggered,
            'breach_time': self.breach_time,
            'remaining_buffer': (self.max_dd - self.current_drawdown) * 100,
            'remaining_buffer_dollars': self.peak_equity * (self.max_dd - self.current_drawdown),
            'update_count': self.update_count
        }
    
    def reset(self, new_capital: Optional[float] = None):
        """
        Reset the drawdown monitor with optional new capital.
        
        This should be used when starting a new trading session or
        after resolving a drawdown breach.
        
        Args:
            new_capital: New starting capital (if None, uses current equity)
            
        Example:
            >>> protector.reset(new_capital=9500)  # Start fresh with remaining capital
        """
        if new_capital is not None:
            if new_capital <= 0:
                raise ValueError(f"new_capital must be positive, got {new_capital}")
            self.initial_capital = new_capital
        else:
            self.initial_capital = self.current_equity
        
        self.peak_equity = self.initial_capital
        self.current_equity = self.initial_capital
        self.current_drawdown = 0.0
        self.breach_triggered = False
        self.breach_time = None
        self.max_drawdown_seen = 0.0
        self.update_count = 0
        
        logger.info(
            f"DrawdownProtector reset with capital=${self.initial_capital:.2f}"
        )
    
    def get_remaining_loss_allowance(self) -> float:
        """
        Calculate how much more loss is allowed before breach.
        
        Returns:
            float: Dollar amount of remaining loss allowance
            
        Example:
            >>> allowance = protector.get_remaining_loss_allowance()
            >>> print(f"Can lose ${allowance:.2f} more before breach")
        """
        max_allowed_loss = self.peak_equity * self.max_dd
        current_loss = self.peak_equity - self.current_equity
        remaining = max_allowed_loss - current_loss
        
        return max(0, remaining)


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage demonstrating drawdown protection in action.
    """
    
    print("\n" + "=" * 70)
    print("DRAWDOWN PROTECTOR - Example Usage")
    print("=" * 70)
    print()
    
    # Initialize protector
    initial_capital = 10000.0
    max_drawdown = 0.05  # 5% max drawdown
    
    protector = DrawdownProtector(
        initial_capital=initial_capital,
        max_dd=max_drawdown
    )
    
    print(f"Initial Setup:")
    print(f"  Capital: ${initial_capital:.2f}")
    print(f"  Max Drawdown Allowed: {max_drawdown*100:.1f}% (${initial_capital * max_drawdown:.2f})")
    print()
    
    # Simulate trading scenarios
    scenarios = [
        # (realized_pnl, unrealized_pnl, description)
        (0, 0, "Start of trading"),
        (100, 50, "Profitable start - up $150"),
        (150, 100, "Continuing profits - up $250"),
        (200, -50, "Some unrealized loss - up $150 net"),
        (150, -200, "Losing position - down $50 net"),
        (100, -300, "Larger losses - down $200 net"),
        (50, -450, "Approaching limit - down $400 net"),
        (0, -550, "BREACH TRIGGERED - down $550 net"),
        (-50, -600, "Continued breach - down $650 net"),
    ]
    
    print("-" * 70)
    print("Trading Simulation:")
    print("-" * 70)
    print()
    
    for i, (realized, unrealized, description) in enumerate(scenarios, 1):
        print(f"Update {i}: {description}")
        print(f"  Realized P&L: ${realized:+.2f}")
        print(f"  Unrealized P&L: ${unrealized:+.2f}")
        
        # Update protector
        is_breach = protector.update(realized, unrealized)
        
        # Get status
        status = protector.get_status()
        
        print(f"  Current Equity: ${status['current_equity']:.2f}")
        print(f"  Peak Equity: ${status['peak_equity']:.2f}")
        print(f"  Current Drawdown: {status['current_drawdown_pct']:.2f}%")
        
        if is_breach:
            print(f"  ⚠️  STATUS: MAX DRAWDOWN BREACHED!")
            print(f"  🛑 ACTION: STOP ALL TRADING IMMEDIATELY")
        else:
            remaining = protector.get_remaining_loss_allowance()
            print(f"  ✓ STATUS: Safe (can lose ${remaining:.2f} more)")
        
        print()
    
    # Final status report
    print("=" * 70)
    print("FINAL STATUS REPORT")
    print("=" * 70)
    
    final_status = protector.get_status()
    
    print(f"\nCapital Management:")
    print(f"  Initial Capital: ${final_status['initial_capital']:.2f}")
    print(f"  Peak Equity: ${final_status['peak_equity']:.2f}")
    print(f"  Current Equity: ${final_status['current_equity']:.2f}")
    print(f"  Total Loss: ${final_status['initial_capital'] - final_status['current_equity']:.2f}")
    
    print(f"\nDrawdown Analysis:")
    print(f"  Current Drawdown: {final_status['current_drawdown_pct']:.2f}%")
    print(f"  Max Drawdown Seen: {final_status['max_drawdown_seen_pct']:.2f}%")
    print(f"  Max Allowed: {final_status['max_allowed_dd_pct']:.2f}%")
    
    print(f"\nBreach Status:")
    if final_status['breach_triggered']:
        print(f"  ⚠️  BREACH OCCURRED")
        print(f"  Time: {final_status['breach_time']}")
        print(f"  Exceeded by: {final_status['current_drawdown_pct'] - final_status['max_allowed_dd_pct']:.2f}%")
    else:
        print(f"  ✓ No breach")
        print(f"  Buffer remaining: {final_status['remaining_buffer']:.2f}%")
        print(f"  (${final_status['remaining_buffer_dollars']:.2f})")
    
    print(f"\nStatistics:")
    print(f"  Updates processed: {final_status['update_count']}")
    
    print()
    print("=" * 70)
    print()
