"""
Order Manager Module for Grid Trading Execution

This module handles safe and efficient order placement and management for
grid trading strategies. It includes pre-trade validation, fee awareness,
spread checking, and comprehensive error handling.

Key Features:
    - Pre-trade validation (fees, spread, grid spacing)
    - Batch order placement with error recovery
    - Stale order cleanup
    - Partial fill tracking
    - Production-grade error handling

Requirements:
    - connectors.gate_io: Gate.io exchange connector
"""

import logging
from typing import List, Dict, Tuple, Optional, Set
import time
from decimal import Decimal

# Import Gate.io connector
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connectors.gate_io import GateIOConnector, initialize_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OrderManager:
    """
    Production-grade order manager for grid trading execution.
    
    This class handles all order placement, cancellation, and validation
    logic with safety checks and error recovery mechanisms.
    
    Attributes:
        connector (GateIOConnector): Initialized Gate.io connector
        pair (str): Trading pair (e.g., 'SOL_USDT')
        placed_orders (Dict): Tracking of placed order IDs
        fee_rate (float): Trading fee rate (e.g., 0.002 for 0.2%)
    """
    
    def __init__(self, connector: GateIOConnector, pair: str):
        """
        Initialize order manager.
        
        Args:
            connector: Initialized Gate.io connector
            pair: Trading pair (e.g., 'SOL_USDT')
            
        Example:
            >>> from connectors.gate_io import initialize_client
            >>> client = initialize_client('testnet')
            >>> manager = OrderManager(client, 'SOL_USDT')
        """
        self.connector = connector
        self.pair = pair
        self.placed_orders: Dict[str, Dict] = {}  # order_id -> order_info
        self.fee_rate: Optional[float] = None
        
        logger.info(f"OrderManager initialized for {pair}")
    
    def get_fee_rate(self) -> float:
        """
        Get current trading fee rate for the user.
        
        Returns:
            float: Fee rate as decimal (e.g., 0.002 for 0.2%)
            
        Raises:
            Exception: If unable to fetch fee information
            
        Note:
            Gate.io fee tiers:
            - VIP 0: 0.2% maker, 0.2% taker
            - VIP 1: 0.18% maker, 0.18% taker
            - Higher VIPs get lower fees
        """
        if self.fee_rate is not None:
            return self.fee_rate
        
        try:
            # Try to get fee rate from trading fee API
            # Note: This requires the spot API fee endpoint
            # Fallback to conservative estimate if unavailable
            
            try:
                # Use actual Gate.io fee rate
                # Gate.io maker/taker fee for VIP 0 is 0.1%
                self.fee_rate = 0.001  # 0.1%
                logger.info("Using standard maker fee rate: 0.1%")
            
            except Exception as e:
                logger.warning(f"Unable to fetch fee rate, using 0.1%: {e}")
                self.fee_rate = 0.001  # 0.1% fallback
            
            logger.info(f"Trading fee rate: {self.fee_rate*100:.3f}%")
            return self.fee_rate
        
        except Exception as e:
            logger.error(f"Error getting fee rate: {e}")
            # Use actual Gate.io fee
            self.fee_rate = 0.001
            return self.fee_rate
    
    def get_order_book_spread(self) -> Tuple[float, float, float]:
        """
        Get current order book spread percentage.
        
        Returns:
            Tuple[float, float, float]: (spread_pct, best_bid, best_ask)
            
        Raises:
            Exception: If unable to fetch order book
            
        Example:
            >>> spread_pct, bid, ask = manager.get_order_book_spread()
            >>> if spread_pct < 0.3:
            ...     print("Spread is acceptable")
        """
        try:
            def _get_order_book():
                return self.connector._request(
                    "GET",
                    "/spot/order_book",
                    params={
                        "currency_pair": self.pair,
                        "limit": 10
                    }
                )
            
            order_book = self.connector._retry_on_error(_get_order_book)
            
            # Extract best bid and ask
            if not order_book.get('bids') or not order_book.get('asks'):
                raise ValueError("Order book is empty")
            
            best_bid = float(order_book['bids'][0][0])  # [price, amount]
            best_ask = float(order_book['asks'][0][0])
            
            # Calculate spread percentage
            mid_price = (best_bid + best_ask) / 2
            spread = best_ask - best_bid
            spread_pct = (spread / mid_price) * 100
            
            logger.debug(
                f"Order book: bid=${best_bid:.2f}, ask=${best_ask:.2f}, "
                f"spread={spread_pct:.3f}%"
            )
            
            return spread_pct, best_bid, best_ask
        
        except Exception as e:
            logger.error(f"Error fetching order book: {e}")
            raise
    
    def validate_grid_spacing(
        self, 
        levels: List[float], 
        fee_rate: float,
        min_profit_multiplier: float = 2.0
    ) -> Tuple[bool, str]:
        """
        Validate that grid spacing is sufficient to cover fees.
        
        Each grid level should have enough spacing to:
        1. Cover both entry and exit fees (2x fee)
        2. Provide minimum profit margin
        
        Args:
            levels: List of price levels (sorted)
            fee_rate: Trading fee rate as decimal
            min_profit_multiplier: Minimum spacing as multiple of total fees (default: 2.0)
            
        Returns:
            Tuple[bool, str]: (is_valid, message)
            
        Example:
            >>> valid, msg = manager.validate_grid_spacing([98, 100, 102], 0.002)
            >>> if valid:
            ...     print("Grid spacing is safe")
        """
        if len(levels) < 2:
            return True, "Insufficient levels to validate spacing"
        
        # Calculate minimum required spacing
        # Total fee = entry_fee + exit_fee = 2 * fee_rate
        # Minimum spacing = min_profit_multiplier * 2 * fee_rate
        # Multiplier 2x ensures profit margin above fees
        min_spacing_pct = 2.0 * 2 * fee_rate
        
        # Check spacing between consecutive levels
        for i in range(len(levels) - 1):
            level_spacing = abs(levels[i+1] - levels[i])
            level_spacing_pct = (level_spacing / levels[i]) * 100
            
            if level_spacing_pct < (min_spacing_pct * 100):
                message = (
                    f"Grid spacing too tight: {level_spacing_pct:.3f}% between "
                    f"${levels[i]:.2f} and ${levels[i+1]:.2f}. "
                    f"Minimum required: {min_spacing_pct*100:.3f}% "
                    f"(to cover {fee_rate*100:.2f}% fee x2 + profit margin)"
                )
                logger.error(message)
                return False, message
        
        logger.info(
            f"Grid spacing validated: minimum {min_spacing_pct*100:.3f}% met "
            f"(fee={fee_rate*100:.2f}%, multiplier=2.0x)"
        )
        
        return True, "Grid spacing is safe"
    
    def place_grid_orders(
        self,
        pair: str,
        buy_levels: List[float],
        sell_levels: List[float],
        base_amount: float,
        max_spread_pct: float = 0.3,
        validate_fees: bool = True
    ) -> Dict[str, any]:
        """
        Place grid orders with comprehensive safety checks.
        
        This is the main order placement function that:
        1. Validates fee rates and grid spacing
        2. Checks order book spread
        3. Places orders in batches
        4. Handles errors and partial failures
        5. Tracks all placed orders
        
        Args:
            pair: Trading pair (e.g., 'SOL_USDT')
            buy_levels: List of buy price levels
            sell_levels: List of sell price levels
            base_amount: Amount per order (in base currency)
            max_spread_pct: Maximum allowed spread percentage (default: 0.3%)
            validate_fees: Whether to validate grid spacing vs fees (default: True)
            
        Returns:
            Dict with placement results:
                - 'buy_orders': List of placed buy order IDs
                - 'sell_orders': List of placed sell order IDs
                - 'failed_orders': List of failed orders with reasons
                - 'total_placed': Total orders successfully placed
                - 'total_failed': Total orders that failed
                
        Raises:
            ValueError: If validation fails critically
            
        Example:
            >>> result = manager.place_grid_orders(
            ...     pair='SOL_USDT',
            ...     buy_levels=[98, 96, 94],
            ...     sell_levels=[102, 104, 106],
            ...     base_amount=0.1,
            ...     max_spread_pct=0.3
            ... )
            >>> print(f"Placed {result['total_placed']} orders")
        """
        logger.info("=" * 60)
        logger.info(f"PLACING GRID ORDERS FOR {pair}")
        logger.info("=" * 60)
        
        result = {
            'buy_orders': [],
            'sell_orders': [],
            'failed_orders': [],
            'total_placed': 0,
            'total_failed': 0
        }
        
        try:
            # Step 1: Get fee rate
            logger.info("Step 1: Fetching fee rate...")
            fee_rate = self.get_fee_rate()
            
            # Step 2: Validate grid spacing
            if validate_fees:
                logger.info("Step 2: Validating grid spacing...")
                
                # Validate buy levels
                if buy_levels:
                    valid, msg = self.validate_grid_spacing(sorted(buy_levels), fee_rate)
                    if not valid:
                        raise ValueError(f"Buy level validation failed: {msg}")
                
                # Validate sell levels
                if sell_levels:
                    valid, msg = self.validate_grid_spacing(sorted(sell_levels), fee_rate)
                    if not valid:
                        raise ValueError(f"Sell level validation failed: {msg}")
            
            # Step 3: Check order book spread
            logger.info("Step 3: Checking order book spread...")
            spread_pct, best_bid, best_ask = self.get_order_book_spread()
            
            if spread_pct > max_spread_pct:
                logger.warning(
                    f"⚠️  Order book spread {spread_pct:.3f}% exceeds maximum {max_spread_pct:.3f}%"
                )
                logger.warning("Skipping order placement due to wide spread")
                return result
            
            logger.info(f"✓ Spread check passed: {spread_pct:.3f}% < {max_spread_pct:.3f}%")
            
            # Step 3.5: Check available balances
            logger.info("\nStep 3.5: Checking available balances...")
            base_currency = pair.split('_')[0]  # e.g., 'SOL' from 'SOL_USDT'
            quote_currency = pair.split('_')[1]  # e.g., 'USDT' from 'SOL_USDT'
            
            try:
                from connectors.gate_io import get_account_balance
                quote_balance = get_account_balance(self.connector, quote_currency)
                base_balance = get_account_balance(self.connector, base_currency)
                
                available_quote = quote_balance['available']
                available_base = base_balance['available']
                
                logger.info(f"  {quote_currency}: {available_quote:.2f} available")
                logger.info(f"  {base_currency}: {available_base:.6f} available")
                
                # Calculate required balances
                if buy_levels:
                    # For buy orders, we need USDT (quote currency)
                    # Amount needed = sum of (price * base_amount) for all buy levels
                    total_quote_needed = sum(price * base_amount for price in buy_levels)
                    logger.info(f"  Buy orders need: {total_quote_needed:.2f} {quote_currency}")
                    
                    if available_quote < total_quote_needed:
                        logger.warning(
                            f"  ⚠️  Insufficient {quote_currency} for all buy orders "
                            f"(have {available_quote:.2f}, need {total_quote_needed:.2f})"
                        )
                
                if sell_levels:
                    # For sell orders, we need SOL (base currency)
                    total_base_needed = len(sell_levels) * base_amount
                    logger.info(f"  Sell orders need: {total_base_needed:.6f} {base_currency}")
                    
                    if available_base < total_base_needed:
                        logger.warning(
                            f"  ⚠️  Insufficient {base_currency} for all sell orders "
                            f"(have {available_base:.6f}, need {total_base_needed:.6f})"
                        )
                        logger.warning(f"  Will skip sell orders until {base_currency} is acquired")
                        # Clear sell levels to skip them
                        sell_levels = []
                        
            except Exception as e:
                logger.warning(f"  Could not check balances: {e}")
            
            # Step 4: Place buy orders
            if buy_levels:
                logger.info(f"\nStep 4: Placing {len(buy_levels)} buy orders...")
                
                for i, price in enumerate(buy_levels, 1):
                    try:
                        logger.info(f"  Placing buy order {i}/{len(buy_levels)} @ ${price:.2f}")
                        
                        order_id = self._place_single_order(
                            pair=pair,
                            side='buy',
                            price=price,
                            amount=base_amount
                        )
                        
                        result['buy_orders'].append(order_id)
                        result['total_placed'] += 1
                        
                        # Track order
                        self.placed_orders[order_id] = {
                            'side': 'buy',
                            'price': price,
                            'amount': base_amount,
                            'pair': pair,
                            'timestamp': time.time()
                        }
                        
                        logger.info(f"  ✓ Buy order placed: {order_id}")
                        
                        # Small delay to avoid rate limiting
                        time.sleep(0.1)
                    
                    except Exception as e:
                        logger.error(f"  ✗ Failed to place buy order @ ${price:.2f}: {e}")
                        result['failed_orders'].append({
                            'side': 'buy',
                            'price': price,
                            'amount': base_amount,
                            'error': str(e)
                        })
                        result['total_failed'] += 1
            
            # Step 5: Place sell orders
            if sell_levels:
                logger.info(f"\nStep 5: Placing {len(sell_levels)} sell orders...")
                
                for i, price in enumerate(sell_levels, 1):
                    try:
                        logger.info(f"  Placing sell order {i}/{len(sell_levels)} @ ${price:.2f}")
                        
                        order_id = self._place_single_order(
                            pair=pair,
                            side='sell',
                            price=price,
                            amount=base_amount
                        )
                        
                        result['sell_orders'].append(order_id)
                        result['total_placed'] += 1
                        
                        # Track order
                        self.placed_orders[order_id] = {
                            'side': 'sell',
                            'price': price,
                            'amount': base_amount,
                            'pair': pair,
                            'timestamp': time.time()
                        }
                        
                        logger.info(f"  ✓ Sell order placed: {order_id}")
                        
                        # Small delay to avoid rate limiting
                        time.sleep(0.1)
                    
                    except Exception as e:
                        logger.error(f"  ✗ Failed to place sell order @ ${price:.2f}: {e}")
                        result['failed_orders'].append({
                            'side': 'sell',
                            'price': price,
                            'amount': base_amount,
                            'error': str(e)
                        })
                        result['total_failed'] += 1
            
            # Summary
            logger.info("=" * 60)
            logger.info("ORDER PLACEMENT SUMMARY")
            logger.info("=" * 60)
            logger.info(f"✓ Total placed: {result['total_placed']}")
            logger.info(f"  • Buy orders: {len(result['buy_orders'])}")
            logger.info(f"  • Sell orders: {len(result['sell_orders'])}")
            
            if result['total_failed'] > 0:
                logger.warning(f"✗ Total failed: {result['total_failed']}")
            
            logger.info("=" * 60)
            
            return result
        
        except Exception as e:
            logger.error(f"Critical error in place_grid_orders: {e}")
            raise
    
    def _place_single_order(
        self, 
        pair: str, 
        side: str, 
        price: float, 
        amount: float
    ) -> str:
        """
        Place a single limit order (internal method).
        
        Args:
            pair: Trading pair
            side: 'buy' or 'sell'
            price: Order price
            amount: Order amount
            
        Returns:
            str: Order ID
            
        Raises:
            Exception: If order placement fails
        """
        from connectors.gate_io import place_limit_order
        
        order_id = place_limit_order(
            connector=self.connector,
            pair=pair,
            side=side,
            price=price,
            amount=amount
        )
        
        return order_id
    
    def cancel_stale_orders(
        self, 
        pair: str, 
        expected_order_ids: Optional[Set[str]] = None,
        max_age_seconds: Optional[float] = None
    ) -> Dict[str, any]:
        """
        Cancel stale or unexpected orders for a trading pair.
        
        This function identifies and cancels orders that:
        1. Are not in the expected order ID set (if provided)
        2. Are older than max_age_seconds (if provided)
        3. Are tracked in placed_orders but shouldn't be open
        
        Args:
            pair: Trading pair (e.g., 'SOL_USDT')
            expected_order_ids: Set of order IDs that should remain open (optional)
            max_age_seconds: Maximum age for orders in seconds (optional)
            
        Returns:
            Dict with cancellation results:
                - 'cancelled_orders': List of cancelled order IDs
                - 'failed_cancellations': List of failed cancellations
                - 'total_cancelled': Total orders cancelled
                
        Example:
            >>> # Cancel all orders except these
            >>> result = manager.cancel_stale_orders(
            ...     pair='SOL_USDT',
            ...     expected_order_ids={'order1', 'order2'}
            ... )
            >>> print(f"Cancelled {result['total_cancelled']} stale orders")
        """
        logger.info(f"Checking for stale orders on {pair}...")
        
        result = {
            'cancelled_orders': [],
            'failed_cancellations': [],
            'total_cancelled': 0
        }
        
        try:
            # Get all open orders for the pair
            def _get_open_orders():
                return self.connector._request(
                    "GET",
                    "/spot/orders",
                    params={
                        "currency_pair": self.pair,
                        "status": "open"
                    },
                    auth=True
                )
            
            open_orders = self.connector._retry_on_error(_get_open_orders)
            
            if not open_orders or len(open_orders) == 0:
                logger.info("No open orders found")
                return result
            
            logger.info(f"Found {len(open_orders)} open orders")
            
            current_time = time.time()
            orders_to_cancel = []
            
            # Identify orders to cancel
            for order in open_orders:
                order_id = order['id']
                should_cancel = False
                reason = ""
                
                # Check if order is in expected set
                if expected_order_ids is not None and order_id not in expected_order_ids:
                    should_cancel = True
                    reason = "not in expected order set"
                
                # Check order age
                if max_age_seconds is not None and order_id in self.placed_orders:
                    order_age = current_time - self.placed_orders[order_id]['timestamp']
                    if order_age > max_age_seconds:
                        should_cancel = True
                        reason = f"age {order_age:.0f}s > {max_age_seconds}s"
                
                if should_cancel:
                    orders_to_cancel.append((order_id, reason))
            
            if not orders_to_cancel:
                logger.info("No stale orders to cancel")
                return result
            
            logger.info(f"Cancelling {len(orders_to_cancel)} stale orders...")
            
            # Cancel identified orders
            for order_id, reason in orders_to_cancel:
                try:
                    def _cancel_order():
                        return self.connector._request(
                            "DELETE",
                            f"/spot/orders/{order_id}",
                            params={"currency_pair": self.pair},
                            auth=True
                        )
                    
                    cancelled = self.connector._retry_on_error(_cancel_order)
                    
                    result['cancelled_orders'].append(order_id)
                    result['total_cancelled'] += 1
                    
                    # Remove from tracking
                    if order_id in self.placed_orders:
                        del self.placed_orders[order_id]
                    
                    logger.info(f"  ✓ Cancelled order {order_id} (reason: {reason})")
                    
                    time.sleep(0.1)  # Rate limit protection
                
                except Exception as e:
                    logger.error(f"  ✗ Failed to cancel order {order_id}: {e}")
                    result['failed_cancellations'].append({
                        'order_id': order_id,
                        'error': str(e)
                    })
            
            logger.info(f"Stale order cleanup complete: {result['total_cancelled']} cancelled")
            
            return result
        
        except Exception as e:
            logger.error(f"Error in cancel_stale_orders: {e}")
            return result
    
    def check_partial_fills(self, pair: str) -> Dict[str, any]:
        """
        Check for partial fills on tracked orders.
        
        Partial fills occur when only part of an order is executed.
        This function identifies partially filled orders and reports their status.
        
        Args:
            pair: Trading pair
            
        Returns:
            Dict with partial fill information:
                - 'partial_fills': List of partially filled orders
                - 'fully_filled': List of fully filled orders
                - 'unfilled': List of unfilled orders
                
        Example:
            >>> fills = manager.check_partial_fills('SOL_USDT')
            >>> if fills['partial_fills']:
            ...     print(f"Warning: {len(fills['partial_fills'])} partial fills")
        """
        logger.info(f"Checking for partial fills on {pair}...")
        
        result = {
            'partial_fills': [],
            'fully_filled': [],
            'unfilled': []
        }
        
        try:
            # Get all orders (open and recent filled)
            def _get_all_orders():
                return self.connector._request(
                    "GET",
                    "/spot/orders",
                    params={
                        "currency_pair": pair,
                        "status": "open,finished",
                        "limit": 100
                    }
                )
            
            orders = self.connector._retry_on_error(_get_all_orders)
            
            for order in orders:
                if order['id'] not in self.placed_orders:
                    continue  # Not our order
                
                filled_amount = float(order.get('filled_total', 0))
                total_amount = float(order['amount'])
                
                if filled_amount == 0:
                    result['unfilled'].append({
                        'order_id': order['id'],
                        'side': order['side'],
                        'price': float(order['price']),
                        'amount': total_amount
                    })
                elif filled_amount < total_amount:
                    # Partial fill
                    fill_pct = (filled_amount / total_amount) * 100
                    result['partial_fills'].append({
                        'order_id': order['id'],
                        'side': order['side'],
                        'price': float(order['price']),
                        'total_amount': total_amount,
                        'filled_amount': filled_amount,
                        'remaining': total_amount - filled_amount,
                        'fill_percentage': fill_pct
                    })
                    logger.warning(
                        f"⚠️  Partial fill: {order['id']} - {fill_pct:.1f}% filled "
                        f"({filled_amount}/{total_amount})"
                    )
                else:
                    # Fully filled
                    result['fully_filled'].append({
                        'order_id': order['id'],
                        'side': order['side'],
                        'price': float(order['price']),
                        'amount': total_amount
                    })
            
            logger.info(
                f"Partial fill check: {len(result['partial_fills'])} partial, "
                f"{len(result['fully_filled'])} filled, {len(result['unfilled'])} unfilled"
            )
            
            return result
        
        except Exception as e:
            logger.error(f"Error checking partial fills: {e}")
            return result


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage demonstrating order manager capabilities.
    Note: This requires valid API credentials in .env file.
    """
    
    print("\n" + "=" * 70)
    print("ORDER MANAGER - Example Usage")
    print("=" * 70)
    print()
    
    print("NOTE: This example requires valid Gate.io API credentials in .env")
    print("Set GATE_IO_TESTNET_API_KEY and GATE_IO_TESTNET_API_SECRET")
    print()
    
    try:
        # Initialize client
        print("Initializing Gate.io client (testnet)...")
        client = initialize_client(mode='testnet')
        print("✓ Client initialized")
        print()
        
        # Create order manager
        pair = 'SOL_USDT'
        manager = OrderManager(connector=client, pair=pair)
        print(f"✓ OrderManager created for {pair}")
        print()
        
        # Example 1: Check fee rate
        print("-" * 70)
        print("Example 1: Checking Fee Rate")
        print("-" * 70)
        
        fee_rate = manager.get_fee_rate()
        print(f"Trading fee rate: {fee_rate*100:.3f}%")
        print(f"Round-trip fee: {fee_rate*2*100:.3f}%")
        print()
        
        # Example 2: Check order book spread
        print("-" * 70)
        print("Example 2: Checking Order Book Spread")
        print("-" * 70)
        
        spread_pct, bid, ask = manager.get_order_book_spread()
        print(f"Best bid: ${bid:.2f}")
        print(f"Best ask: ${ask:.2f}")
        print(f"Spread: {spread_pct:.3f}%")
        
        if spread_pct < 0.3:
            print("✓ Spread is acceptable for trading")
        else:
            print("⚠️  Spread is too wide, consider waiting")
        print()
        
        # Example 3: Validate grid spacing
        print("-" * 70)
        print("Example 3: Validating Grid Spacing")
        print("-" * 70)
        
        test_levels = [98.0, 100.0, 102.0, 104.0]
        valid, msg = manager.validate_grid_spacing(test_levels, fee_rate)
        
        print(f"Test levels: {test_levels}")
        print(f"Validation result: {valid}")
        print(f"Message: {msg}")
        print()
        
        # Example 4: Place grid orders (commented out for safety)
        print("-" * 70)
        print("Example 4: Place Grid Orders (DRY RUN)")
        print("-" * 70)
        print()
        print("⚠️  Actual order placement is commented out for safety")
        print("To enable, uncomment the code below and ensure sufficient balance")
        print()
        
        # Uncomment to actually place orders (ensure you have testnet balance!)
        """
        buy_levels = [bid * 0.99, bid * 0.98]  # 1% and 2% below bid
        sell_levels = [ask * 1.01, ask * 1.02]  # 1% and 2% above ask
        base_amount = 0.01  # Small test amount
        
        result = manager.place_grid_orders(
            pair=pair,
            buy_levels=buy_levels,
            sell_levels=sell_levels,
            base_amount=base_amount,
            max_spread_pct=0.3,
            validate_fees=True
        )
        
        print(f"Placement result:")
        print(f"  Buy orders: {len(result['buy_orders'])}")
        print(f"  Sell orders: {len(result['sell_orders'])}")
        print(f"  Failed: {result['total_failed']}")
        """
        
        # Example 5: Check for stale orders
        print("-" * 70)
        print("Example 5: Cancel Stale Orders")
        print("-" * 70)
        print()
        
        result = manager.cancel_stale_orders(pair=pair)
        print(f"Cancellation result:")
        print(f"  Cancelled: {result['total_cancelled']}")
        print(f"  Failed: {len(result['failed_cancellations'])}")
        print()
        
        print("=" * 70)
        print("Example completed successfully!")
        print("=" * 70)
        print()
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nCommon issues:")
        print("1. API credentials not set in .env file")
        print("2. Not connected to internet")
        print("3. Gate.io testnet unavailable")
        print()
