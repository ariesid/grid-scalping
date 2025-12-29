"""
Trade Logger Module for CSV Export

Tracks all trades and profits in CSV format for analysis and reporting.
"""

import csv
import os
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class TradeLogger:
    """
    Logs trade execution details and profits to CSV file.
    
    CSV Columns:
        - timestamp: ISO format datetime
        - iteration: Bot iteration number
        - order_id: Exchange order ID
        - side: buy/sell
        - price: Execution price
        - amount: Order amount
        - value_usdt: Total value in USDT
        - fee_usdt: Trading fee in USDT
        - profit_usdt: Profit from this trade (for sells)
        - cumulative_profit: Running total profit
        - equity: Current total equity
        - notes: Additional information
    """
    
    def __init__(self, csv_path: str = "trades.csv", initial_capital: float = 0):
        """
        Initialize trade logger.
        
        Args:
            csv_path: Path to CSV file
            initial_capital: Starting capital for profit calculation
        """
        self.csv_path = csv_path
        self.initial_capital = initial_capital
        self.cumulative_profit = 0.0
        self.buy_prices: Dict[str, float] = {}  # Track buy prices for profit calc
        
        # Create CSV with headers if it doesn't exist
        if not os.path.exists(csv_path):
            self._create_csv()
            logger.info(f"📊 Trade log initialized: {csv_path}")
        else:
            # Load existing cumulative profit
            self._load_state()
            logger.info(f"📊 Trade log loaded: {csv_path} (cumulative profit: ${self.cumulative_profit:.2f})")
    
    def _create_csv(self) -> None:
        """Create CSV file with headers."""
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp',
                'iteration',
                'order_id',
                'side',
                'price',
                'amount',
                'value_usdt',
                'fee_usdt',
                'profit_usdt',
                'cumulative_profit',
                'equity',
                'notes'
            ])
    
    def _load_state(self) -> None:
        """Load cumulative profit from existing CSV."""
        try:
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    last_row = rows[-1]
                    self.cumulative_profit = float(last_row.get('cumulative_profit', 0))
                    
                    # Rebuild buy_prices from history
                    for row in rows:
                        if row['side'] == 'buy' and row['order_id']:
                            self.buy_prices[row['order_id']] = float(row['price'])
        except Exception as e:
            logger.warning(f"Could not load trade log state: {e}")
            self.cumulative_profit = 0.0
    
    def log_order_placed(
        self,
        iteration: int,
        order_id: str,
        side: str,
        price: float,
        amount: float,
        notes: str = ""
    ) -> None:
        """
        Log when an order is placed (not filled yet).
        
        Args:
            iteration: Bot iteration number
            order_id: Exchange order ID
            side: 'buy' or 'sell'
            price: Order price
            amount: Order amount
            notes: Additional notes
        """
        # Store buy price for later profit calculation
        if side == 'buy':
            self.buy_prices[order_id] = price
    
    def log_order_filled(
        self,
        iteration: int,
        order_id: str,
        side: str,
        price: float,
        amount: float,
        fee_rate: float = 0.002,
        equity: float = 0,
        notes: str = ""
    ) -> float:
        """
        Log a filled order and calculate profit.
        
        Args:
            iteration: Bot iteration number
            order_id: Exchange order ID
            side: 'buy' or 'sell'
            price: Fill price
            amount: Fill amount
            fee_rate: Trading fee rate (default 0.2%)
            equity: Current total equity
            notes: Additional notes
            
        Returns:
            Profit from this trade (0 for buys, calculated for sells)
        """
        value_usdt = price * amount
        fee_usdt = value_usdt * fee_rate
        profit_usdt = 0.0
        
        # Calculate profit for sell orders
        if side == 'sell':
            # Find matching buy price (simplified - using average if multiple)
            buy_price = self.buy_prices.get(order_id, price * 0.99)  # Default to small profit
            buy_value = buy_price * amount
            buy_fee = buy_value * fee_rate
            
            # Profit = (sell_value - sell_fee) - (buy_value + buy_fee)
            profit_usdt = (value_usdt - fee_usdt) - (buy_value + buy_fee)
            self.cumulative_profit += profit_usdt
            
            # Remove from tracking
            if order_id in self.buy_prices:
                del self.buy_prices[order_id]
        
        # Write to CSV
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                iteration,
                order_id,
                side,
                f"{price:.8f}",
                f"{amount:.8f}",
                f"{value_usdt:.4f}",
                f"{fee_usdt:.4f}",
                f"{profit_usdt:.4f}",
                f"{self.cumulative_profit:.4f}",
                f"{equity:.4f}",
                notes
            ])
        
        # Log to console
        if side == 'sell' and profit_usdt != 0:
            profit_emoji = "💰" if profit_usdt > 0 else "📉"
            logger.info(
                f"{profit_emoji} Order {order_id} filled: {side.upper()} {amount:.4f} @ ${price:.2f} "
                f"→ Profit: ${profit_usdt:+.4f} (Total: ${self.cumulative_profit:+.4f})"
            )
        else:
            logger.info(f"📝 Order {order_id} filled: {side.upper()} {amount:.4f} @ ${price:.2f}")
        
        return profit_usdt
    
    def get_summary(self) -> Dict:
        """
        Get trading summary statistics.
        
        Returns:
            Dict with summary statistics
        """
        try:
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
                if not rows:
                    return {
                        'total_trades': 0,
                        'total_profit': 0.0,
                        'roi_pct': 0.0,
                        'buy_count': 0,
                        'sell_count': 0
                    }
                
                buy_count = sum(1 for r in rows if r['side'] == 'buy')
                sell_count = sum(1 for r in rows if r['side'] == 'sell')
                total_profit = self.cumulative_profit
                roi_pct = (total_profit / self.initial_capital * 100) if self.initial_capital > 0 else 0
                
                return {
                    'total_trades': len(rows),
                    'total_profit': total_profit,
                    'roi_pct': roi_pct,
                    'buy_count': buy_count,
                    'sell_count': sell_count,
                    'avg_profit_per_trade': total_profit / sell_count if sell_count > 0 else 0
                }
        except Exception as e:
            logger.error(f"Error getting trade summary: {e}")
            return {
                'total_trades': 0,
                'total_profit': self.cumulative_profit,
                'roi_pct': 0.0,
                'buy_count': 0,
                'sell_count': 0
            }
    
    def print_summary(self) -> None:
        """Print formatted trade summary to console."""
        summary = self.get_summary()
        
        logger.info("\n" + "=" * 70)
        logger.info("📊 TRADE SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total Trades: {summary['total_trades']} ({summary['buy_count']} buys, {summary['sell_count']} sells)")
        logger.info(f"Cumulative Profit: ${summary['total_profit']:+.4f}")
        logger.info(f"ROI: {summary['roi_pct']:+.2f}%")
        if summary['sell_count'] > 0:
            logger.info(f"Avg Profit/Trade: ${summary['avg_profit_per_trade']:+.4f}")
        logger.info(f"CSV Log: {self.csv_path}")
        logger.info("=" * 70)
