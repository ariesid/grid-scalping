"""
Gate.io Connector Module for SOL/USDT Scalping Grid Bot

This module provides a unified interface to interact with Gate.io exchange
for spot trading operations. It supports both testnet and mainnet modes,
handles rate limiting, and includes retry logic for robust API communication.

Requirements:
    - requests: For HTTP API calls
    - python-dotenv: For loading environment variables
"""

import os
import time
import logging
import json
import hmac
import hashlib
from typing import Optional, Dict, Any
from decimal import Decimal

import requests
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Global configuration
GATE_IO_CONFIG = {
    'mainnet': {
        'host': 'https://api.gateio.ws',
        'api_key_env': 'GATE_IO_API_KEY',
        'api_secret_env': 'GATE_IO_API_SECRET'
    },
    'testnet': {
        'host': 'https://api-testnet.gateapi.io',
        'api_key_env': 'GATE_IO_TESTNET_API_KEY',
        'api_secret_env': 'GATE_IO_TESTNET_API_SECRET'
    }
}

# API prefix
API_PREFIX = '/api/v4'

# Rate limiting and retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds
RATE_LIMIT_DELAY = 2  # seconds


def _sha512_hex(s: str) -> str:
    """Generate SHA512 hex digest of a string."""
    return hashlib.sha512((s or "").encode("utf-8")).hexdigest()


class GateIOConnector:
    """
    Gate.io API connector with automatic retry and rate limit handling.
    
    Attributes:
        mode (str): Trading mode - 'mainnet' or 'testnet'
        host (str): API host URL
        api_key (str): API key
        api_secret (str): API secret
        session (requests.Session): HTTP session for requests
    """
    
    def __init__(self, mode: str = 'testnet'):
        """
        Initialize Gate.io connector.
        
        Args:
            mode (str): Trading mode - 'mainnet' or 'testnet'. Defaults to 'testnet'.
            
        Raises:
            ValueError: If mode is invalid or API credentials are missing.
        """
        if mode not in GATE_IO_CONFIG:
            raise ValueError(f"Invalid mode: {mode}. Must be 'mainnet' or 'testnet'")
        
        self.mode = mode
        self.config = GATE_IO_CONFIG[mode]
        self.host = self.config['host']
        
        # Get credentials
        self.api_key, self.api_secret = self._get_credentials()
        
        # Setup HTTP session
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json"
        })
        
        logger.info(f"Initializing Gate.io connector in {mode} mode")
    
    def _get_credentials(self) -> tuple:
        """
        Retrieve API credentials from environment variables.
        
        Returns:
            tuple: (api_key, api_secret)
            
        Raises:
            ValueError: If credentials are not found in environment.
        """
        api_key = os.getenv(self.config['api_key_env'])
        api_secret = os.getenv(self.config['api_secret_env'])
        
        if not api_key or not api_secret:
            raise ValueError(
                f"Missing API credentials. Please set {self.config['api_key_env']} "
                f"and {self.config['api_secret_env']} in your .env file"
            )
        
        return api_key, api_secret
    
    def _sign_headers(self, method: str, url_path: str, query_str: str, body_str: str) -> dict:
        """
        Generate authentication headers for Gate.io API request.
        
        Args:
            method: HTTP method (GET, POST, DELETE, etc.)
            url_path: API endpoint path
            query_str: Query string parameters
            body_str: Request body as JSON string
            
        Returns:
            dict: Headers with KEY, Timestamp, and SIGN
        """
        ts = str(int(time.time()))
        sign_str = f"{method}\n{API_PREFIX}{url_path}\n{query_str}\n{_sha512_hex(body_str)}\n{ts}"
        sign = hmac.new(
            self.api_secret.encode("utf-8"),
            sign_str.encode("utf-8"),
            hashlib.sha512
        ).hexdigest()
        
        return {
            "KEY": self.api_key,
            "Timestamp": ts,
            "SIGN": sign
        }
    
    def _request(
        self,
        method: str,
        url_path: str,
        params: Optional[dict] = None,
        body: Optional[dict] = None,
        auth: bool = False
    ) -> Any:
        """
        Make HTTP request to Gate.io API.
        
        Args:
            method: HTTP method
            url_path: API endpoint path
            params: Query parameters
            body: Request body
            auth: Whether to include authentication headers
            
        Returns:
            JSON response data
            
        Raises:
            RuntimeError: If request fails
        """
        query_str = ""
        if params:
            query_str = "&".join(f"{k}={v}" for k, v in params.items())
        
        body_str = json.dumps(body) if body else ""
        
        headers = {}
        if auth:
            headers = self._sign_headers(method, url_path, query_str, body_str)
        
        url = f"{self.host}{API_PREFIX}{url_path}"
        if query_str:
            url += f"?{query_str}"
        
        try:
            r = self.session.request(
                method,
                url,
                headers=headers,
                data=body_str if body else None,
                timeout=15
            )
            
            if r.status_code >= 400:
                raise RuntimeError(f"{r.status_code} | {r.text}")
            
            return r.json()
        
        except requests.exceptions.Timeout:
            raise RuntimeError("Request timeout")
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(f"Connection error: {e}")
        except Exception as e:
            raise RuntimeError(f"Request error: {e}")
    
    def _retry_on_error(self, func, *args, **kwargs):
        """
        Execute function with retry logic for rate limits and transient errors.
        
        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Function result
            
        Raises:
            Exception: If all retries are exhausted
        """
        last_exception = None
        
        for attempt in range(MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            
            except RuntimeError as e:
                last_exception = e
                error_msg = str(e)
                
                # Handle rate limiting (HTTP 429)
                if '429' in error_msg:
                    wait_time = RATE_LIMIT_DELAY * (attempt + 1)
                    logger.warning(
                        f"Rate limit hit. Waiting {wait_time}s before retry "
                        f"(attempt {attempt + 1}/{MAX_RETRIES})"
                    )
                    time.sleep(wait_time)
                    continue
                
                # Handle server errors (5xx)
                elif any(f'{code}' in error_msg for code in range(500, 600)):
                    wait_time = RETRY_DELAY * (attempt + 1)
                    logger.warning(
                        f"Server error. Retrying in {wait_time}s "
                        f"(attempt {attempt + 1}/{MAX_RETRIES})"
                    )
                    time.sleep(wait_time)
                    continue
                
                # For other errors, don't retry
                else:
                    logger.error(f"API error: {error_msg}")
                    raise
            
            except Exception as e:
                last_exception = e
                wait_time = RETRY_DELAY * (attempt + 1)
                logger.warning(
                    f"Unexpected error: {str(e)}. Retrying in {wait_time}s "
                    f"(attempt {attempt + 1}/{MAX_RETRIES})"
                )
                time.sleep(wait_time)
                continue
        
        # All retries exhausted
        logger.error(f"All {MAX_RETRIES} retries exhausted")
        raise last_exception


def initialize_client(mode: str = 'testnet') -> GateIOConnector:
    """
    Initialize and configure Gate.io API client.
    
    Args:
        mode (str): Trading mode - 'mainnet' or 'testnet'. Defaults to 'testnet'.
        
    Returns:
        GateIOConnector: Configured connector instance with authenticated API client.
        
    Raises:
        ValueError: If mode is invalid or API credentials are missing.
        
    Example:
        >>> client = initialize_client('testnet')
        >>> price = get_ticker_price(client, 'SOL_USDT')
    """
    connector = GateIOConnector(mode)
    logger.info(f"Gate.io client initialized successfully in {mode} mode")
    return connector


def get_ticker_price(connector: GateIOConnector, pair: str) -> float:
    """
    Get current ticker price for a trading pair.
    
    Args:
        connector (GateIOConnector): Initialized Gate.io connector
        pair (str): Trading pair (e.g., 'SOL_USDT')
        
    Returns:
        float: Current last trade price
        
    Raises:
        RuntimeError: If API request fails
        ValueError: If ticker data is invalid
        
    Example:
        >>> client = initialize_client('testnet')
        >>> price = get_ticker_price(client, 'SOL_USDT')
        >>> print(f"Current SOL price: ${price}")
    """
    def _fetch_ticker():
        data = connector._request("GET", "/spot/tickers", params={"currency_pair": pair})
        if not data or len(data) == 0:
            raise ValueError(f"No ticker data found for {pair}")
        return data[0]
    
    try:
        ticker = connector._retry_on_error(_fetch_ticker)
        price = float(ticker["last"])
        logger.info(f"Fetched ticker price for {pair}: {price}")
        return price
    
    except Exception as e:
        logger.error(f"Failed to fetch ticker price for {pair}: {str(e)}")
        raise


def place_limit_order(
    connector: GateIOConnector,
    pair: str,
    side: str,
    price: float,
    amount: float
) -> str:
    """
    Place a limit order on Gate.io spot market.
    
    Args:
        connector (GateIOConnector): Initialized Gate.io connector
        pair (str): Trading pair (e.g., 'SOL_USDT')
        side (str): Order side - 'buy' or 'sell'
        price (float): Limit price
        amount (float): Order amount (in base currency)
        
    Returns:
        str: Order ID
        
    Raises:
        RuntimeError: If order placement fails
        ValueError: If parameters are invalid
        
    Example:
        >>> client = initialize_client('testnet')
        >>> order_id = place_limit_order(client, 'SOL_USDT', 'buy', 100.50, 0.5)
        >>> print(f"Order placed: {order_id}")
    """
    if side not in ['buy', 'sell']:
        raise ValueError(f"Invalid side: {side}. Must be 'buy' or 'sell'")
    
    if price <= 0 or amount <= 0:
        raise ValueError(f"Price and amount must be positive. Got price={price}, amount={amount}")
    
    # Create order payload
    body = {
        "currency_pair": pair,
        "side": side,
        "type": "limit",
        "amount": str(amount),
        "price": str(price),
        "time_in_force": "gtc"  # Good Till Cancelled
    }
    
    def _place_order():
        return connector._request("POST", "/spot/orders", body=body, auth=True)
    
    try:
        response = connector._retry_on_error(_place_order)
        order_id = response.get("id")
        logger.info(f"Order placed: {side} {amount} {pair} @ {price}, ID: {order_id}")
        return order_id
    
    except Exception as e:
        logger.error(f"Failed to place order: {str(e)}")
        raise


def get_open_orders(connector: GateIOConnector, pair: str) -> list:
    """
    Get all open orders for a trading pair.
    
    Args:
        connector (GateIOConnector): Initialized Gate.io connector
        pair (str): Trading pair (e.g., 'SOL_USDT')
        
    Returns:
        list: List of open order dictionaries
        
    Example:
        >>> client = initialize_client('testnet')
        >>> orders = get_open_orders(client, 'SOL_USDT')
        >>> print(f"Open orders: {len(orders)}")
    """
    def _get_orders():
        return connector._request(
            "GET",
            "/spot/orders",
            params={
                "currency_pair": pair,
                "status": "open"
            },
            auth=True
        )
    
    try:
        orders = connector._retry_on_error(_get_orders)
        return orders if orders else []
    except Exception as e:
        logger.error(f"Failed to fetch open orders for {pair}: {str(e)}")
        return []


def get_order_status(connector: GateIOConnector, pair: str, order_id: str) -> Optional[Dict]:
    """
    Get detailed status of a specific order.
    
    Args:
        connector (GateIOConnector): Initialized Gate.io connector
        pair (str): Trading pair (e.g., 'SOL_USDT')
        order_id (str): Order ID to check
        
    Returns:
        dict: Order details including status ('open', 'closed', 'cancelled')
        None: If order not found
        
    Example:
        >>> client = initialize_client('testnet')
        >>> order = get_order_status(client, 'SOL_USDT', '12345')
        >>> if order and order['status'] == 'closed':
        >>>     print(f"Order filled at {order['fill_price']}")
    """
    def _get_order():
        return connector._request(
            "GET",
            f"/spot/orders/{order_id}",
            params={"currency_pair": pair},
            auth=True
        )
    
    try:
        order = connector._retry_on_error(_get_order)
        return order
    except Exception as e:
        logger.debug(f"Failed to fetch order {order_id}: {str(e)}")
        return None


def cancel_orders_by_distance(
    connector: GateIOConnector, 
    pair: str, 
    current_price: float,
    max_distance_pct: float = 2.0
) -> Dict[str, int]:
    """
    Cancel orders that are too far from current price.
    
    Smart cancellation that only removes orders unlikely to fill soon:
    - Sell orders > max_distance_pct above current price
    - Buy orders > max_distance_pct below current price
    - Keeps orders that are still in reasonable range to fill
    
    Args:
        connector: Initialized Gate.io connector
        pair: Trading pair (e.g., 'SOL_USDT')
        current_price: Current market price
        max_distance_pct: Maximum distance % to keep orders (default: 2.0%)
        
    Returns:
        dict: {'cancelled': count, 'kept': count, 'details': list}
        
    Example:
        >>> # Current price $124.50
        >>> # Will cancel: sell @ $127.50 (2.4% away)
        >>> # Will keep: sell @ $125.50 (0.8% away)
        >>> result = cancel_orders_by_distance(client, 'SOL_USDT', 124.50, 2.0)
        >>> print(f"Cancelled {result['cancelled']}, kept {result['kept']}")
    """
    result = {'cancelled': 0, 'kept': 0, 'details': []}
    
    try:
        # Get all open orders
        open_orders = get_open_orders(connector, pair)
        
        if not open_orders:
            logger.debug("No open orders to check")
            return result
        
        logger.info(f"Checking {len(open_orders)} orders against current price ${current_price:.2f}")
        
        for order in open_orders:
            order_id = order['id']
            order_price = float(order['price'])
            order_side = order['side']
            
            # Calculate distance from current price
            if order_side == 'sell':
                # Sell order: check if price is too far above current
                distance_pct = ((order_price - current_price) / current_price) * 100
            else:  # buy
                # Buy order: check if price is too far below current
                distance_pct = ((current_price - order_price) / current_price) * 100
            
            # Cancel if too far away
            if distance_pct > max_distance_pct:
                try:
                    def _cancel():
                        return connector._request(
                            "DELETE",
                            f"/spot/orders/{order_id}",
                            params={"currency_pair": pair},
                            auth=True
                        )
                    connector._retry_on_error(_cancel)
                    result['cancelled'] += 1
                    result['details'].append({
                        'order_id': order_id,
                        'side': order_side,
                        'price': order_price,
                        'distance_pct': distance_pct,
                        'action': 'cancelled'
                    })
                    logger.info(f"  ✗ Cancelled {order_side} @ ${order_price:.2f} ({distance_pct:+.2f}% from current)")
                    time.sleep(0.1)
                except Exception as e:
                    logger.error(f"  Failed to cancel order {order_id}: {e}")
            else:
                result['kept'] += 1
                result['details'].append({
                    'order_id': order_id,
                    'side': order_side,
                    'price': order_price,
                    'distance_pct': distance_pct,
                    'action': 'kept'
                })
                logger.debug(f"  ✓ Keeping {order_side} @ ${order_price:.2f} ({distance_pct:+.2f}% from current)")
        
        logger.info(f"Order cleanup: {result['cancelled']} cancelled, {result['kept']} kept")
        return result
        
    except Exception as e:
        logger.error(f"Error in smart order cancellation: {e}")
        return result


def cancel_all_orders(connector: GateIOConnector, pair: str) -> int:
    """
    Cancel all open orders for a trading pair.
    
    Args:
        connector (GateIOConnector): Initialized Gate.io connector
        pair (str): Trading pair (e.g., 'SOL_USDT')
        
    Returns:
        int: Number of orders cancelled
        
    Raises:
        RuntimeError: If cancellation fails
        
    Example:
        >>> client = initialize_client('testnet')
        >>> cancelled = cancel_all_orders(client, 'SOL_USDT')
        >>> print(f"Cancelled {cancelled} orders")
    """
    def _cancel_orders():
        return connector._request(
            "DELETE",
            "/spot/orders",
            params={"currency_pair": pair},
            auth=True
        )
    
    try:
        # Cancel all orders
        cancelled_orders = connector._retry_on_error(_cancel_orders)
        cancelled_count = len(cancelled_orders) if isinstance(cancelled_orders, list) else 0
        
        logger.info(f"Cancelled {cancelled_count} orders for {pair}")
        return cancelled_count
    
    except Exception as e:
        logger.error(f"Failed to cancel orders for {pair}: {str(e)}")
        raise


def get_account_balance(connector: GateIOConnector, currency: str = 'USDT') -> Dict[str, float]:
    """
    Get account balance for a specific currency.
    
    Args:
        connector (GateIOConnector): Initialized Gate.io connector
        currency (str): Currency code (e.g., 'USDT', 'SOL')
        
    Returns:
        dict: Balance information with 'available' and 'locked' amounts
        
    Raises:
        RuntimeError: If balance fetch fails
        
    Example:
        >>> client = initialize_client('testnet')
        >>> balance = get_account_balance(client, 'USDT')
        >>> print(f"Available: {balance['available']} USDT")
    """
    def _fetch_balance():
        accounts = connector._request("GET", "/spot/accounts", params={"currency": currency}, auth=True)
        if not accounts or len(accounts) == 0:
            return {'available': 0.0, 'locked': 0.0}
        
        account = accounts[0]
        return {
            'available': float(account.get('available', 0)),
            'locked': float(account.get('locked', 0))
        }
    
    try:
        balance = connector._retry_on_error(_fetch_balance)
        logger.info(
            f"Account balance for {currency}: "
            f"Available={balance['available']}, Locked={balance['locked']}"
        )
        return balance
    
    except Exception as e:
        logger.error(f"Failed to fetch balance for {currency}: {str(e)}")
        raise


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of the Gate.io connector module.
    Make sure to set up your .env file with the required API credentials.
    """
    
    # Example .env file structure:
    # GATE_IO_TESTNET_API_KEY=your_testnet_api_key
    # GATE_IO_TESTNET_API_SECRET=your_testnet_api_secret
    # GATE_IO_API_KEY=your_mainnet_api_key
    # GATE_IO_API_SECRET=your_mainnet_api_secret
    
    try:
        # Initialize client in testnet mode
        client = initialize_client(mode='testnet')
        
        # Get current SOL/USDT price
        pair = 'SOL_USDT'
        current_price = get_ticker_price(client, pair)
        print(f"Current {pair} price: ${current_price}")
        
        # Get account balance
        usdt_balance = get_account_balance(client, 'USDT')
        print(f"USDT Balance - Available: {usdt_balance['available']}, Locked: {usdt_balance['locked']}")
        
        # Place a test limit buy order (adjust price to ensure it doesn't fill immediately)
        test_price = current_price * 0.95  # 5% below current price
        test_amount = 0.01  # Small test amount
        
        # Uncomment to actually place an order
        # order_id = place_limit_order(client, pair, 'buy', test_price, test_amount)
        # print(f"Test order placed: {order_id}")
        
        # Cancel all open orders
        # cancelled = cancel_all_orders(client, pair)
        # print(f"Cancelled {cancelled} orders")
        
    except Exception as e:
        logger.error(f"Error in example usage: {str(e)}")
        raise
