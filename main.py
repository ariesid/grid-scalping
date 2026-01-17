"""
SOL/USDT Grid Scalping Bot - Main Orchestrator

This is the main entry point for the grid trading bot. It orchestrates all
modules and manages the trading loop, risk monitoring, and state persistence.

Features:
    - Automated grid trading on SOL/USDT
    - Real-time trend and volatility analysis
    - Risk management (drawdown protection, trailing stops)
    - State persistence and recovery
    - Graceful shutdown handling

Usage:
    python main.py --mode testnet
    python main.py --mode mainnet --pair SOL_USDT --capital 10000
"""

import argparse
import logging
import os
import signal
import sys
import time
from datetime import datetime
from typing import Optional, Dict, List
import pandas as pd

# Import all bot modules
from connectors.gate_io import initialize_client, get_ticker_price
from market.indicators import calculate_adx, calculate_atr, calculate_ema
from strategy.grid_engine import generate_grid_levels
from strategy.trend_filter import can_activate_grid, analyze_market_conditions
from risk.drawdown_monitor import DrawdownProtector
from risk.trailing_stop import TrailingStop
from execution.order_manager import OrderManager
from recovery.state_sync import StateManager
from utils.trade_logger import TradeLogger

# Configure logging
# Clear any existing handlers and configure from scratch
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Remove any existing handlers
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add file handler
file_handler = logging.FileHandler('grid_bot.log', mode='a', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
root_logger.addHandler(file_handler)

# Add console handler with UTF-8 encoding for Windows
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
# Set encoding to UTF-8 to handle special characters
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)


class GridScalpingBot:
    """
    Main bot orchestrator for SOL/USDT grid scalping strategy.
    
    This class manages all bot operations including:
    - Market data fetching
    - Trend analysis and filtering
    - Grid level generation and order placement
    - Risk monitoring and protection
    - State persistence and recovery
    
    Attributes:
        mode (str): Trading mode ('testnet' or 'mainnet')
        pair (str): Trading pair (e.g., 'SOL_USDT')
        initial_capital (float): Starting capital
        loop_interval (int): Main loop interval in seconds
        is_running (bool): Bot running state
    """
    
    def __init__(
        self,
        mode: str = 'testnet',
        pair: str = 'SOL_USDT',
        initial_capital: float = 10000.0,
        loop_interval: int = 20,
        max_drawdown: float = 0.01,
        trailing_stop_pct: float = 0.015
    ):
        """
        Initialize the grid scalping bot.
        
        Args:
            mode: Trading mode - 'testnet' or 'mainnet' (default: 'testnet')
            pair: Trading pair (default: 'SOL_USDT')
            initial_capital: Starting capital (default: 10000.0)
            loop_interval: Loop interval in seconds (default: 20)
            max_drawdown: Maximum allowed drawdown (default: 0.01 for 1%)
            trailing_stop_pct: Trailing stop percentage (default: 0.015 for 1.5%)
        """
        logger.info("=" * 70)
        logger.info("INITIALIZING GRID SCALPING BOT")
        logger.info("=" * 70)
        
        self.mode = mode
        self.pair = pair
        self.initial_capital = initial_capital
        self.loop_interval = loop_interval
        self.trailing_stop_pct = trailing_stop_pct
        self.is_running = False
        
        # Bot state
        self.grid_active = False
        self.current_orders: List[str] = []
        self.iteration_count = 0
        self.last_grid_update = 0
        self.last_grid_price = 0.0  # Track price when grid was last placed
        self.last_order_count = {}  # Track order counts for fill detection
        
        # Initialize modules
        logger.info(f"\nMode: {mode.upper()}")
        logger.info(f"Trading Pair: {pair}")
        logger.info(f"Initial Capital: ${initial_capital:.2f}")
        logger.info(f"Loop Interval: {loop_interval}s")
        logger.info("")
        
        try:
            # 1. Initialize Gate.io connector
            logger.info("1. Initializing Gate.io connector...")
            self.connector = initialize_client(mode=mode)
            logger.info("   ✓ Connector initialized")
            
            # 2. Initialize order manager
            logger.info("2. Initializing order manager...")
            self.order_manager = OrderManager(self.connector, pair)
            logger.info("   ✓ Order manager initialized")
            
            # 3. Initialize drawdown protector
            logger.info("3. Initializing drawdown protector...")
            self.drawdown_protector = DrawdownProtector(
                initial_capital=initial_capital,
                max_dd=max_drawdown
            )
            logger.info(f"   ✓ Drawdown protector initialized (max: {max_drawdown*100:.1f}%)")
            
            # 4. Initialize trailing stop (will be created when grid becomes active)
            logger.info(f"4. Trailing stop ready (trail: {trailing_stop_pct*100:.1f}%)")
            self.trailing_stop: Optional[TrailingStop] = None
            self.position_avg_price: Optional[float] = None  # Track average entry price
            
            # 5. Initialize state manager
            logger.info("5. Initializing state manager...")
            self.state_manager = StateManager('bot_state.json')
            logger.info("   ✓ State manager initialized")
            
            # 6. Initialize trade logger
            logger.info("6. Initializing trade logger...")
            self.trade_logger = TradeLogger(
                csv_path="trades.csv",
                initial_capital=initial_capital
            )
            logger.info("   ✓ Trade logger initialized")
            
            # 7. Try to recover previous state
            logger.info("7. Checking for previous state...")
            self._recover_state()
            
            logger.info("\n" + "=" * 70)
            logger.info("✓ BOT INITIALIZATION COMPLETE")
            logger.info("=" * 70)
            logger.info("")
            
        except Exception as e:
            logger.error(f"Failed to initialize bot: {e}")
            raise
    
    def _recover_state(self) -> None:
        """Attempt to recover from previous state."""
        try:
            state = self.state_manager.load_state(
                connector=self.connector,
                pair=self.pair,
                reconcile=True
            )
            
            if state:
                logger.info("   Previous state found - recovering...")
                logger.info(f"   State from: {state['timestamp']}")
                logger.info(f"   Last equity: ${state['last_equity']:.2f}")
                
                # Check reconciliation
                if 'reconciliation' in state:
                    recon = state['reconciliation']
                    logger.info(f"   Matched orders: {len(recon['matched_orders'])}")
                    logger.info(f"   Missing orders: {len(recon['missing_orders'])}")
                    
                    if self.state_manager.needs_grid_rebuild(recon):
                        logger.warning("   ⚠️  Grid rebuild needed - will regenerate")
                        self.grid_active = False
                    else:
                        logger.info("   ✓ State recovered successfully")
                        self.current_orders = recon['matched_orders']
                        self.grid_active = len(self.current_orders) > 0
            else:
                logger.info("   No previous state found - starting fresh")
                
        except Exception as e:
            logger.warning(f"   Could not recover state: {e}")
            logger.info("   Starting with clean state")
    
    def fetch_ohlcv_data(self, lookback_candles: int = 100) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for analysis.
        
        Args:
            lookback_candles: Number of historical candles to fetch
            
        Returns:
            DataFrame with OHLCV data or None if fetch fails
        """
        try:
            # Fetch candlestick data from Gate.io
            def _get_candles():
                return self.connector._request(
                    "GET",
                    "/spot/candlesticks",
                    params={
                        "currency_pair": self.pair,
                        "interval": "1m",
                        "limit": lookback_candles
                    }
                )
            
            candles = self.connector._retry_on_error(_get_candles)
            
            if not candles:
                logger.warning("No candlestick data received")
                return None
            
            # Convert to DataFrame
            data = []
            for candle in candles:
                data.append({
                    'timestamp': int(candle[0]),
                    'volume': float(candle[1]),
                    'close': float(candle[2]),
                    'high': float(candle[3]),
                    'low': float(candle[4]),
                    'open': float(candle[5])
                })
            
            df = pd.DataFrame(data)
            df = df.sort_values('timestamp')
            
            logger.debug(f"Fetched {len(df)} candles")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch OHLCV data: {e}")
            return None
    
    def check_trend_conditions(self, df: pd.DataFrame) -> bool:
        """
        Check if market conditions allow grid trading.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            bool: True if conditions are favorable, False if should pause
        """
        try:
            # For tight scalping in 121-122 range, allow slightly higher ADX
            # ADX 28 = weak trend, still acceptable for tight range scalping
            # Relaxed RSI bounds (20-80) to allow grid trading in more conditions
            can_activate = can_activate_grid(
                df,
                adx_threshold=28.0,  # Increased from 25 for scalping
                rsi_lower_bound=20.0,  # Relaxed from 30 - allow slightly oversold
                rsi_upper_bound=80.0,  # Relaxed from 70 - allow slightly overbought
                max_ema_distance_pct=5.0
            )
            
            if not can_activate:
                logger.warning("⚠️  Market conditions unfavorable - pausing grid")
                return False
            
            logger.info("✓ Market conditions favorable for grid trading")
            return True
            
        except Exception as e:
            logger.error(f"Error checking trend conditions: {e}")
            return False
    
    def update_risk_monitors(self) -> bool:
        """
        Update risk monitoring systems.
        
        Returns:
            bool: True if risk checks pass, False if breach detected
        """
        try:
            # Get current balances
            from connectors.gate_io import get_account_balance
            
            usdt_balance = get_account_balance(self.connector, 'USDT')
            sol_balance = get_account_balance(self.connector, 'SOL')
            
            # Calculate current equity including both available AND locked funds
            # Locked funds are in open orders and still part of your capital
            current_price = get_ticker_price(self.connector, self.pair)
            
            # Total USDT = available + locked (in buy orders)
            total_usdt = usdt_balance['available'] + usdt_balance['locked']
            
            # Total SOL value = (available + locked) * current price
            total_sol = sol_balance['available'] + sol_balance['locked']
            sol_value_usdt = total_sol * current_price
            
            # Current equity = all USDT + SOL valued in USDT
            current_equity = total_usdt + sol_value_usdt
            
            # Calculate P&L
            unrealized_pnl = 0  # No position-based unrealized for grid trading
            realized_pnl = current_equity - self.initial_capital
            
            # Update drawdown monitor
            is_breach = self.drawdown_protector.update(realized_pnl, unrealized_pnl)
            
            if is_breach:
                logger.critical("🛑 MAX DRAWDOWN BREACHED - STOPPING BOT")
                return False
            
            # Log risk status
            status = self.drawdown_protector.get_status()
            logger.info(
                f"Risk Monitor: Drawdown={status['current_drawdown_pct']:.2f}%, "
                f"Equity=${status['current_equity']:.2f}"
            )
            
            # Trailing stop logic (only when grid is active)
            if self.grid_active:
                # Initialize trailing stop if not already created
                if self.trailing_stop is None:
                    self.trailing_stop = TrailingStop(side='long')
                    # Use last grid price as position entry price
                    self.position_avg_price = self.last_grid_price
                    logger.info(f"✓ Trailing stop activated at ${self.position_avg_price:.2f}")
                
                # Update trailing stop with current price
                if self.position_avg_price:
                    self.trailing_stop.update(current_price, self.position_avg_price)
                    
                    # Check if trailing stop exit signal triggered
                    if self.trailing_stop.should_exit(trail_pct=self.trailing_stop_pct):
                        logger.warning("🛑 TRAILING STOP TRIGGERED - EXITING ALL POSITIONS")
                        
                        # Cancel all orders
                        from connectors.gate_io import cancel_all_orders
                        cancelled = cancel_all_orders(self.connector, self.pair)
                        logger.info(f"Cancelled {cancelled} orders")
                        
                        # Deactivate grid and reset trailing stop
                        self.grid_active = False
                        self.current_orders = []
                        self.trailing_stop = None
                        self.position_avg_price = None
                        
                        logger.info("Grid deactivated - waiting for new entry conditions")
            else:
                # Reset trailing stop when grid is not active
                if self.trailing_stop is not None:
                    self.trailing_stop = None
                    self.position_avg_price = None
            
            # Check for filled orders and log to CSV
            self._check_filled_orders(status['current_equity'])
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating risk monitors: {e}")
            return True  # Continue on error (fail-safe)
    
    def _check_filled_orders(self, equity: float) -> None:
        """Check for filled orders and log to CSV."""
        try:
            from connectors.gate_io import get_open_orders, get_order_status
            
            # Get current open orders
            open_orders = get_open_orders(self.connector, self.pair)
            current_order_ids = {order['id'] for order in open_orders}
            
            # Check if any tracked orders are now filled
            for order_id in list(self.order_manager.placed_orders.keys()):
                if order_id not in current_order_ids:
                    # Order is no longer open - check if it was filled or cancelled
                    order_status = get_order_status(self.connector, self.pair, order_id)
                    
                    # Only log if order was actually filled (status = 'closed')
                    # Don't log cancelled orders
                    if order_status and order_status.get('status') == 'closed':
                        order_info = self.order_manager.placed_orders.get(order_id)
                        if order_info:
                            logger.info(f"🎯 Order {order_id} FILLED: {order_info['side']} @ ${order_info['price']:.2f}")
                            self.trade_logger.log_order_filled(
                                iteration=self.iteration_count,
                                order_id=order_id,
                                side=order_info['side'],
                                price=order_info['price'],
                                amount=order_info['amount'],
                                fee_rate=0.001,  # 0.1% Gate.io fee
                                equity=equity,
                                notes=f"Filled at iteration {self.iteration_count}"
                            )
                    elif order_status and order_status.get('status') == 'cancelled':
                        logger.debug(f"Order {order_id} was cancelled, not filled")
                    
                    # Remove from tracking regardless of status
                    if order_id in self.order_manager.placed_orders:
                        del self.order_manager.placed_orders[order_id]
                        
        except Exception as e:
            logger.debug(f"Could not check filled orders: {e}")
    
    def generate_and_place_grid(self, df: pd.DataFrame) -> bool:
        """
        Generate grid levels and place orders.
        
        Args:
            df: OHLCV DataFrame for analysis
            
        Returns:
            bool: True if grid placed successfully
        """
        try:
            logger.info("Generating grid levels...")
            
            # Get current price
            current_price = get_ticker_price(self.connector, self.pair)
            logger.info(f"Current price: ${current_price:.2f}")
            
            # Calculate ATR for adaptive spacing
            atr = calculate_atr(
                df['high'].tolist(),
                df['low'].tolist(),
                df['close'].tolist(),
                period=14
            )
            logger.info(f"ATR(14): ${atr:.2f}")
            
            # Analyze market to determine trend bias
            analysis = analyze_market_conditions(df)
            
            if analysis['adx'] >= 30:
                trend_bias = 'bullish' if analysis['price_vs_ema']['position'] == 'above' else 'bearish'
            else:
                trend_bias = 'neutral'
            
            logger.info(f"Trend bias: {trend_bias}")
            
            # Generate grid levels
            # SOL/USDT scalping setup: Dynamic 0.83% range centered on current price
            # With 8 levels: ~0.12% spacing per level (~$0.15 at $123)
            grid = generate_grid_levels(
                current_price=current_price,
                atr_value=atr,
                grid_range_pct=0.0083,  # 0.83% range (dynamic around current price)
                total_levels=8,
                trend_bias=trend_bias,
                atr_multiplier=0.3  # Tight spacing for scalping
            )
            
            logger.info(
                f"Grid generated: {len(grid['buy_levels'])} buy levels, "
                f"{len(grid['sell_levels'])} sell levels"
            )
            
            # Place grid orders
            base_amount = 0.025  # Meets Gate.io $3 minimum (0.025 SOL × $120 = $3)
            
            result = self.order_manager.place_grid_orders(
                pair=self.pair,
                buy_levels=grid['buy_levels'],
                sell_levels=grid['sell_levels'],
                base_amount=base_amount,
                max_spread_pct=0.3,
                validate_fees=True
            )
            
            if result['total_placed'] > 0:
                # Log newly placed orders to trade logger
                all_new_orders = result['buy_orders'] + result['sell_orders']
                logger.info(f"Logging {len(all_new_orders)} orders to CSV...")
                for order_id in all_new_orders:
                    if order_id in self.order_manager.placed_orders:
                        order_info = self.order_manager.placed_orders[order_id]
                        logger.info(f"  Logging order {order_id}: {order_info['side']} @ ${order_info['price']:.2f}")
                        self.trade_logger.log_order_placed(
                            iteration=self.iteration_count,
                            order_id=order_id,
                            side=order_info['side'],
                            price=order_info['price'],
                            amount=order_info['amount'],
                            notes="Grid order placed"
                        )
                logger.info(f"✓ CSV logging complete")
            
            if result['total_placed'] > 0:
                logger.info(f"✓ Grid placed: {result['total_placed']} orders")
                self.current_orders = result['buy_orders'] + result['sell_orders']
                self.grid_active = True
                self.last_grid_update = time.time()
                self.last_grid_price = current_price  # Track price when grid was placed
                return True
            else:
                logger.warning("No orders placed")
                return False
            
        except Exception as e:
            logger.error(f"Error generating/placing grid: {e}")
            return False
    
    def heartbeat_check(self) -> None:
        """Run health check and log bot status."""
        try:
            status = {
                'timestamp': datetime.now().isoformat(),
                'iteration': self.iteration_count,
                'mode': self.mode,
                'pair': self.pair,
                'grid_active': self.grid_active,
                'active_orders': len(self.current_orders),
                'uptime_minutes': (time.time() - self.start_time) / 60
            }
            
            # Get current equity
            dd_status = self.drawdown_protector.get_status()
            status['equity'] = dd_status['current_equity']
            status['drawdown_pct'] = dd_status['current_drawdown_pct']
            
            logger.info("=" * 70)
            logger.info("HEARTBEAT CHECK")
            logger.info("=" * 70)
            logger.info(f"Iteration: {status['iteration']}")
            logger.info(f"Uptime: {status['uptime_minutes']:.1f} minutes")
            logger.info(f"Grid Active: {status['grid_active']}")
            logger.info(f"Active Orders: {status['active_orders']}")
            logger.info(f"Equity: ${status['equity']:.2f}")
            logger.info(f"Drawdown: {status['drawdown_pct']:.2f}%")
            logger.info("=" * 70)
            
        except Exception as e:
            logger.error(f"Error in heartbeat check: {e}")
    
    def save_current_state(self) -> None:
        """Save current bot state to disk."""
        try:
            # Get current balances
            from connectors.gate_io import get_account_balance
            
            balances = {
                'USDT': get_account_balance(self.connector, 'USDT')['available'],
                'SOL': get_account_balance(self.connector, 'SOL')['available']
            }
            
            # Get active orders info
            active_orders = []
            for order_id in self.current_orders:
                if order_id in self.order_manager.placed_orders:
                    active_orders.append(self.order_manager.placed_orders[order_id])
            
            # Grid parameters (simplified)
            grid_parameters = {
                'pair': self.pair,
                'grid_active': self.grid_active,
                'last_update': self.last_grid_update
            }
            
            # Get equity
            dd_status = self.drawdown_protector.get_status()
            
            # Save state
            self.state_manager.save_state(
                balances=balances,
                active_orders=active_orders,
                grid_parameters=grid_parameters,
                last_equity=dd_status['current_equity'],
                metadata={
                    'mode': self.mode,
                    'iteration': self.iteration_count,
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            logger.info("✓ State saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def run(self) -> None:
        """
        Main bot loop.
        
        This is the core execution loop that runs continuously until stopped.
        """
        logger.info("\n" + "=" * 70)
        logger.info("STARTING BOT MAIN LOOP")
        logger.info("=" * 70)
        logger.info(f"Loop interval: {self.loop_interval} seconds")
        logger.info("Press Ctrl+C to stop gracefully")
        logger.info("=" * 70)
        logger.info("")
        
        self.is_running = True
        self.start_time = time.time()
        
        try:
            while self.is_running:
                self.iteration_count += 1
                iteration_start = time.time()
                
                logger.info(f"{'='*70}")
                logger.info(f"ITERATION {self.iteration_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"{'='*70}\n")
                
                try:
                    # Step 1: Fetch latest price & OHLCV
                    logger.info("Step 1: Fetching market data...")
                    current_price = get_ticker_price(self.connector, self.pair)
                    logger.info(f"  Current price: ${current_price:.2f}")
                    
                    df = self.fetch_ohlcv_data(lookback_candles=100)
                    if df is None or len(df) < 50:
                        logger.warning("  Insufficient OHLCV data - skipping iteration")
                        time.sleep(self.loop_interval)
                        continue
                    
                    logger.info(f"  ✓ Fetched {len(df)} candles")
                    
                    # Step 2: Check trend conditions
                    logger.info("\nStep 2: Checking trend conditions...")
                    conditions_ok = self.check_trend_conditions(df)
                    
                    if not conditions_ok:
                        if self.grid_active:
                            logger.warning("  ⚠️  Conditions unfavorable - pausing NEW order placement")
                            logger.info("  📊 Existing orders remain active and will be monitored for fills")
                            # Set grid to inactive to prevent placing new orders
                            # But DON'T cancel existing orders - let them fill
                            self.grid_active = False
                        else:
                            # Even when grid inactive, check for filled orders periodically
                            if self.iteration_count % 5 == 0:
                                self._check_filled_orders(self.drawdown_protector.current_equity)
                    
                    # Step 3: Update risk monitors
                    logger.info("\nStep 3: Updating risk monitors...")
                    risk_ok = self.update_risk_monitors()
                    
                    if not risk_ok:
                        logger.critical("  Risk breach detected - stopping bot")
                        self.stop()
                        break
                    
                    # Step 4: Place/update grid if conditions allow
                    logger.info("\nStep 4: Grid management...")
                    
                    if conditions_ok and not self.grid_active:
                        logger.info("  Conditions favorable - activating grid")
                        success = self.generate_and_place_grid(df)
                        if success:
                            logger.info("  ✓ Grid activated")
                    elif self.grid_active:
                        logger.info("  ✓ Grid already active")
                        
                        # Check for filled orders
                        self._check_filled_orders(self.drawdown_protector.current_equity)
                        
                        # Calculate price movement since last grid placement
                        price_move_pct = abs(current_price - self.last_grid_price) / self.last_grid_price * 100
                        
                        # Smart order cleanup: cancel orders too far from current price
                        # This keeps orders that can still fill but removes unreachable ones
                        if self.iteration_count % 10 == 0:  # Check every 10 iterations (~3 minutes)
                            from connectors.gate_io import cancel_orders_by_distance
                            result = cancel_orders_by_distance(
                                self.connector,
                                self.pair,
                                current_price,
                                max_distance_pct=2.0  # Cancel orders >2% away from current price
                            )
                            if result['cancelled'] > 0:
                                logger.info(f"  Cleaned up {result['cancelled']} orders too far from price")
                        
                        # Regenerate grid only if price moved significantly (>0.5%)
                        if price_move_pct > 0.5:
                            logger.info(f"  Price moved {price_move_pct:.2f}% - regenerating grid")
                            # Cancel all current orders before regenerating
                            from connectors.gate_io import cancel_all_orders
                            cancel_all_orders(self.connector, self.pair)
                            self.grid_active = False
                            self.current_orders = []
                            # Will regenerate on next iteration
                    else:
                        logger.info("  Grid inactive (waiting for conditions)")
                        # Still check for filled orders even when paused
                        self._check_filled_orders(self.drawdown_protector.current_equity)
                        # Also clean up orders too far from current price
                        if self.iteration_count % 10 == 0:
                            from connectors.gate_io import cancel_orders_by_distance
                            result = cancel_orders_by_distance(
                                self.connector,
                                self.pair,
                                current_price,
                                max_distance_pct=2.5  # Slightly more tolerance when paused
                            )
                            if result['cancelled'] > 0:
                                logger.info(f"  Cleaned up {result['cancelled']} orders too far from price")
                    
                    # Step 5: Heartbeat check (every 10 iterations)
                    if self.iteration_count % 10 == 0:
                        logger.info("\nStep 5: Running heartbeat check...")
                        self.heartbeat_check()
                    
                    # Save state periodically (every 5 iterations)
                    if self.iteration_count % 5 == 0:
                        logger.info("\nSaving state...")
                        self.save_current_state()
                    
                except Exception as e:
                    logger.error(f"Error in iteration {self.iteration_count}: {e}")
                    logger.exception(e)
                
                # Calculate sleep time
                iteration_time = time.time() - iteration_start
                sleep_time = max(0, self.loop_interval - iteration_time)
                
                logger.info(f"\n✓ Iteration {self.iteration_count} complete ({iteration_time:.1f}s)")
                logger.info(f"Sleeping for {sleep_time:.1f}s...\n")
                
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("\n\nKeyboard interrupt received")
            self.stop()
        except Exception as e:
            logger.error(f"Fatal error in main loop: {e}")
            logger.exception(e)
            self.stop()
    
    def stop(self) -> None:
        """
        Gracefully stop the bot.
        
        This handles cleanup, state saving, and shutdown procedures.
        """
        if not self.is_running:
            return
        
        logger.info("\n" + "=" * 70)
        logger.info("INITIATING GRACEFUL SHUTDOWN")
        logger.info("=" * 70)
        
        self.is_running = False
        
        try:
            # 1. Save final state
            logger.info("\n1. Saving final state...")
            self.save_current_state()
            
            # 2. Cancel all orders (optional - comment out to leave orders active)
            # logger.info("\n2. Cancelling all open orders...")
            # from connectors.gate_io import cancel_all_orders
            # cancelled = cancel_all_orders(self.connector, self.pair)
            # logger.info(f"   Cancelled {cancelled} orders")
            
            # 3. Log final statistics
            logger.info("\n3. Final statistics...")
            final_status = self.drawdown_protector.get_status()
            
            logger.info(f"   Total iterations: {self.iteration_count}")
            logger.info(f"   Final equity: ${final_status['current_equity']:.2f}")
            logger.info(f"   Total P&L: ${final_status['current_equity'] - self.initial_capital:.2f}")
            logger.info(f"   Max drawdown seen: {final_status['max_drawdown_seen_pct']:.2f}%")
            
            # 4. Print trade summary from CSV
            logger.info("\n4. Trade summary...")
            self.trade_logger.print_summary()
            
            logger.info("\n" + "=" * 70)
            logger.info("✓ SHUTDOWN COMPLETE")
            logger.info("=" * 70)
            logger.info("")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


def setup_signal_handlers(bot: GridScalpingBot) -> None:
    """
    Set up signal handlers for graceful shutdown.
    
    Args:
        bot: Bot instance to shutdown on signal
    """
    def signal_handler(signum, frame):
        logger.info(f"\nReceived signal {signum}")
        bot.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='SOL/USDT Grid Scalping Bot',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode testnet
  python main.py --mode mainnet --pair SOL_USDT --capital 10000
  python main.py --mode testnet --interval 30 --max-drawdown 0.03
        """
    )
    
    # Load defaults from .env file
    from dotenv import load_dotenv
    load_dotenv()
    
    default_mode = os.getenv('TRADING_MODE', 'testnet')
    default_pair = os.getenv('TRADING_PAIR', 'SOL_USDT')
    
    parser.add_argument(
        '--mode',
        type=str,
        default=default_mode,
        choices=['testnet', 'mainnet'],
        help=f'Trading mode (default: {default_mode} from .env)'
    )
    
    parser.add_argument(
        '--pair',
        type=str,
        default=default_pair,
        help=f'Trading pair (default: {default_pair} from .env)'
    )
    
    parser.add_argument(
        '--capital',
        type=float,
        default=10000.0,
        help='Initial capital (default: 10000.0)'
    )
    
    parser.add_argument(
        '--interval',
        type=int,
        default=20,
        help='Loop interval in seconds (default: 20)'
    )
    
    parser.add_argument(
        '--max-drawdown',
        type=float,
        default=0.05,
        help='Maximum allowed drawdown (default: 0.05 for 5%%)'
    )
    
    parser.add_argument(
        '--trailing-stop',
        type=float,
        default=0.015,
        help='Trailing stop percentage (default: 0.015 for 1.5%%)'
    )
    
    args = parser.parse_args()
    
    # Display startup banner
    print("\n" + "=" * 70)
    print("SOL/USDT GRID SCALPING BOT")
    print("=" * 70)
    print(f"Mode: {args.mode.upper()}")
    print(f"Pair: {args.pair}")
    print(f"Capital: ${args.capital:.2f}")
    print(f"Loop Interval: {args.interval}s")
    print(f"Max Drawdown: {args.max_drawdown*100:.1f}%")
    print(f"Trailing Stop: {args.trailing_stop*100:.1f}%")
    print("=" * 70)
    print("")
    
    if args.mode == 'mainnet':
        print("⚠️  WARNING: Running in MAINNET mode with REAL MONEY!")
        print("Press Ctrl+C within 10 seconds to cancel...")
        try:
            time.sleep(10)
        except KeyboardInterrupt:
            print("\nCancelled by user")
            sys.exit(0)
    
    try:
        # Initialize bot
        bot = GridScalpingBot(
            mode=args.mode,
            pair=args.pair,
            initial_capital=args.capital,
            loop_interval=args.interval,
            max_drawdown=args.max_drawdown,
            trailing_stop_pct=args.trailing_stop
        )
        
        # Set up signal handlers
        setup_signal_handlers(bot)
        
        # Run bot
        bot.run()
        
    except Exception as e:
        logger.error(f"Failed to start bot: {e}")
        logger.exception(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
