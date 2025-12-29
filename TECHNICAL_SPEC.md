# Technical Specification: SOL/USDT Grid Scalping Bot

**Version**: 1.0  
**Last Updated**: December 29, 2025  
**Language**: Python 3.8+  
**Exchange**: Gate.io (Spot Trading)

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Module Specifications](#module-specifications)
4. [Algorithms](#algorithms)
5. [Data Structures](#data-structures)
6. [API Interfaces](#api-interfaces)
7. [State Management](#state-management)
8. [Error Handling](#error-handling)
9. [Performance Considerations](#performance-considerations)
10. [Extension Points](#extension-points)

---

## System Overview

### Purpose

Automated grid trading bot for cryptocurrency scalping on Gate.io exchange. Implements adaptive grid levels with intelligent market condition filtering and comprehensive risk management.

### Key Design Principles

- **Modularity**: Clear separation of concerns across functional modules
- **Resilience**: Graceful error handling and state recovery
- **Observability**: Comprehensive logging and trade tracking
- **Safety**: Multi-layered risk protection mechanisms

### Technology Stack

- **Core**: Python 3.8+
- **HTTP Client**: `requests` library
- **Data Processing**: `pandas`, `numpy`
- **Configuration**: Environment variables via `python-dotenv`
- **Persistence**: JSON file storage

---

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Main Orchestrator                       │
│                      (main.py)                               │
│  • Initialization                                            │
│  • Main Trading Loop                                         │
│  • Signal Handling                                           │
│  • Shutdown Management                                       │
└───────────────┬────────────────────────────────┬─────────────┘
                │                                │
    ┌───────────▼─────────┐          ┌──────────▼──────────┐
    │   Market Analysis   │          │   Order Execution   │
    │                     │          │                     │
    │  • Trend Filter     │          │  • Order Manager    │
    │  • Grid Engine      │          │  • Balance Check    │
    │  • Indicators       │          │  • Validation       │
    └───────────┬─────────┘          └──────────┬──────────┘
                │                                │
    ┌───────────▼─────────┐          ┌──────────▼──────────┐
    │  Risk Management    │          │   State/Logging     │
    │                     │          │                     │
    │  • Drawdown Monitor │          │  • State Manager    │
    │  • Trailing Stop    │          │  • Trade Logger     │
    └─────────────────────┘          └─────────────────────┘
                │                                │
    ┌───────────▼──────────────────────────────▼──────────┐
    │              Exchange Connector                       │
    │              (connectors/gate_io.py)                 │
    │  • API Authentication                                │
    │  • Rate Limiting                                     │
    │  • Error Retry Logic                                 │
    └──────────────────────────────────────────────────────┘
```

### Data Flow

```
1. Market Data Fetch → 2. Trend Analysis → 3. Grid Generation →
4. Risk Check → 5. Order Placement → 6. State Persistence →
7. Fill Detection → 8. P&L Update → [Loop]
```

---

## Module Specifications

### 1. Main Orchestrator (`main.py`)

**Class**: `GridScalpingBot`

**Responsibilities**:

- Initialize all subsystems
- Execute main trading loop
- Coordinate between modules
- Handle graceful shutdown

**Key Methods**:

```python
__init__(mode, pair, initial_capital, loop_interval, max_drawdown)
    # Initialize bot with configuration parameters

run()
    # Main trading loop execution
    # Returns: None
    # Raises: KeyboardInterrupt on Ctrl+C

_update_risk_monitors() -> bool
    # Update drawdown protection and check limits
    # Returns: True if safe to continue, False if breach

_check_filled_orders(equity: float)
    # Check for filled orders and log to CSV
    # Updates internal state

_graceful_shutdown()
    # Save state and cleanup on exit
```

**Main Loop Logic**:

```
1. Fetch market data (price, candles)
2. Check trend conditions (ADX, RSI, EMA)
3. Update risk monitors (drawdown, equity)
4. If conditions favorable:
   a. Generate grid levels
   b. Place orders (skip if insufficient balance)
   c. Log placed orders
5. Check for filled orders
6. Save state
7. Sleep for interval
8. Repeat
```

---

### 2. Exchange Connector (`connectors/gate_io.py`)

**Class**: `GateIOConnector`

**Authentication**: HMAC-SHA512 signature

```python
sign_str = f"{method}\n{url_path}\n{query_string}\n{sha512(body)}\n{timestamp}"
signature = hmac_sha512(api_secret, sign_str)
```

**API Endpoints Used**:

- `GET /api/v4/spot/accounts` - Get account balances
- `GET /api/v4/spot/tickers` - Get ticker prices
- `GET /api/v4/spot/candlesticks` - Get OHLCV candles
- `GET /api/v4/spot/order_book` - Get order book
- `POST /api/v4/spot/orders` - Place order
- `GET /api/v4/spot/orders/{order_id}` - Get order status
- `GET /api/v4/spot/open_orders` - Get open orders
- `DELETE /api/v4/spot/orders/{order_id}` - Cancel order

**Rate Limiting**:

- Built-in retry logic with exponential backoff
- Configurable delays between requests
- Max retries: 3

**Error Handling**:

```python
try:
    response = self._request(method, endpoint, params)
except RequestError as e:
    # Log error
    # Retry with backoff
    # Raise if max retries exceeded
```

---

### 3. Market Indicators (`market/indicators.py`)

**Functions**:

```python
calculate_ema(prices: np.ndarray, period: int) -> float
    # Exponential Moving Average
    # Formula: EMA = (Price × K) + (EMA_prev × (1 - K))
    # where K = 2 / (period + 1)

calculate_atr(df: pd.DataFrame, period: int = 14) -> float
    # Average True Range (volatility measure)
    # TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
    # ATR = EMA(TR, period)

calculate_adx(df: pd.DataFrame, period: int = 14) -> float
    # Average Directional Index (trend strength)
    # +DI = EMA(+DM, period) / ATR
    # -DI = EMA(-DM, period) / ATR
    # DX = 100 × |+DI - -DI| / (+DI + -DI)
    # ADX = EMA(DX, period)

calculate_rsi(prices: np.ndarray, period: int = 14) -> float
    # Relative Strength Index (momentum)
    # RS = Average Gain / Average Loss
    # RSI = 100 - (100 / (1 + RS))
```

**Data Requirements**:

- Minimum candles for ADX: 2 × period + 1 (e.g., 29 for period=14)
- Minimum candles for EMA: period (e.g., 50 for EMA50)
- Minimum candles for RSI: period + 1 (e.g., 15 for period=14)

---

### 4. Trend Filter (`strategy/trend_filter.py`)

**Function**: `can_activate_grid()`

**Decision Logic**:

```python
def can_activate_grid(
    df: pd.DataFrame,
    current_price: float,
    adx_threshold: float = 28.0,
    rsi_lower_bound: float = 20.0,
    rsi_upper_bound: float = 80.0,
    max_ema_distance_pct: float = 5.0
) -> bool:

    # Calculate indicators
    adx = calculate_adx(df)
    rsi = calculate_rsi(df['close'])
    ema50 = calculate_ema(df['close'], 50)

    # Check conditions
    is_ranging = adx < adx_threshold
    is_rsi_neutral = rsi_lower_bound < rsi < rsi_upper_bound
    is_near_ema = abs(current_price - ema50) / ema50 < max_ema_distance_pct / 100

    # All conditions must pass
    return is_ranging and is_rsi_neutral and is_near_ema
```

**Condition Explanations**:

1. **ADX < 28**: Ranging market (not trending)
2. **RSI 20-80**: Neutral momentum (not overbought/oversold)
3. **Price within 5% of EMA50**: Near dynamic support/resistance

---

### 5. Grid Engine (`strategy/grid_engine.py`)

**Class**: `GridEngine`

**Grid Generation Algorithm**:

```python
def generate_grid_levels(
    current_price: float,
    atr: float,
    grid_range_pct: float = 0.10,  # ±10% range
    num_levels: int = 5,
    trend_bias: str = "neutral",
    atr_multiplier: float = 0.5
) -> Dict[str, List[float]]:

    # Calculate bounds
    lower_bound = current_price * (1 - grid_range_pct / 2)
    upper_bound = current_price * (1 + grid_range_pct / 2)

    # ATR-based step size
    step = atr * atr_multiplier

    # Generate levels
    buy_levels = []
    sell_levels = []

    # Buy levels below current price
    price = current_price - step
    while price >= lower_bound and len(buy_levels) < num_levels:
        buy_levels.append(price)
        price -= step

    # Sell levels above current price
    price = current_price + step
    while price <= upper_bound and len(sell_levels) < num_levels:
        sell_levels.append(price)
        price += step

    return {
        'buy_levels': sorted(buy_levels, reverse=True),  # Highest to lowest
        'sell_levels': sorted(sell_levels)  # Lowest to highest
    }
```

**Minimum Spacing Enforcement**:

```python
min_spacing_pct = 0.01  # 1% minimum
if abs(level2 - level1) / level1 < min_spacing_pct:
    # Adjust or skip level
```

---

### 6. Order Manager (`execution/order_manager.py`)

**Class**: `OrderManager`

**Order Placement Flow**:

```
1. Get fee rate (0.1% for Gate.io)
2. Validate grid spacing (must cover 2× fee + margin)
3. Check order book spread (< 0.3% max)
4. Check available balances
5. Place buy orders (requires USDT)
6. Place sell orders (requires SOL)
7. Track placed orders
8. Return placement results
```

**Fee Validation**:

```python
min_spacing = 2 × 2 × fee_rate  # 2× (buy+sell), 2× (multiplier)
# For 0.1% fee: 0.4% minimum spacing

for i in range(len(levels) - 1):
    spacing_pct = abs(levels[i+1] - levels[i]) / levels[i] × 100
    if spacing_pct < min_spacing × 100:
        raise ValueError("Grid spacing too tight")
```

**Balance Check Logic**:

```python
# Check USDT for buy orders
total_usdt_needed = sum(price × amount for price in buy_levels)
if available_usdt < total_usdt_needed:
    log_warning("Insufficient USDT")

# Check SOL for sell orders
total_sol_needed = len(sell_levels) × amount
if available_sol < total_sol_needed:
    log_warning("Insufficient SOL - skipping sell orders")
    sell_levels = []  # Clear to skip
```

---

### 7. Drawdown Monitor (`risk/drawdown_monitor.py`)

**Class**: `DrawdownProtector`

**Equity Calculation**:

```python
# Total USDT (available + locked in orders)
total_usdt = usdt_balance['available'] + usdt_balance['locked']

# Total SOL value
total_sol = sol_balance['available'] + sol_balance['locked']
sol_value_usdt = total_sol × current_price

# Total equity
current_equity = total_usdt + sol_value_usdt
```

**Drawdown Calculation**:

```python
# Update peak if current is higher
if current_equity > peak_equity:
    peak_equity = current_equity

# Calculate drawdown
drawdown = (peak_equity - current_equity) / peak_equity

# Check breach
if drawdown > max_drawdown:
    trigger_protection()
    return True  # Breach detected
```

**Protection Actions**:

1. Set `breach_triggered = True`
2. Record `breach_time`
3. Return to main loop
4. Main loop stops trading and initiates shutdown

---

### 8. State Manager (`recovery/state_sync.py`)

**Class**: `StateManager`

**State Structure**:

```json
{
  "version": "1.0",
  "timestamp": "2025-12-29T10:30:00.000000",
  "balances": {
    "USDT": 59.61,
    "SOL": 0.001152
  },
  "active_orders": [
    {
      "side": "buy",
      "price": 124.14,
      "amount": 0.025,
      "pair": "SOL_USDT",
      "timestamp": 1735463400.123
    }
  ],
  "grid_parameters": {
    "pair": "SOL_USDT",
    "grid_active": true,
    "last_update": 1735463400.456
  },
  "last_equity": 59.76,
  "metadata": {
    "mode": "mainnet",
    "iteration": 42,
    "timestamp": "2025-12-29T10:30:00.000000"
  }
}
```

**State Operations**:

```python
save_state()
    # Write state to bot_state.json
    # Create backup with timestamp
    # Cleanup old backups (keep last 10)

load_state() -> Optional[Dict]
    # Read bot_state.json
    # Validate version and structure
    # Return None if not found or invalid

restore_orders()
    # Re-populate order tracking from state
    # Verify orders still open on exchange
```

**Backup Management**:

- Location: `backups/bot_state_backup_YYYYMMDD_HHMMSS.json`
- Retention: Last 10 backups
- Frequency: Every state save

---

### 9. Trade Logger (`utils/trade_logger.py`)

**Class**: `TradeLogger`

**CSV Format**:

```csv
timestamp,iteration,order_id,side,price,amount,value_usdt,fee_usdt,profit_usdt,cumulative_profit,equity,notes
2025-12-29T10:30:00,42,123456789,buy,124.14,0.025,3.10,0.0031,0.0000,0.0000,59.76,Filled at iteration 42
2025-12-29T10:35:00,45,987654321,sell,124.82,0.025,3.12,0.0031,0.0168,0.0168,59.93,Filled at iteration 45
```

**Profit Calculation**:

```python
# For buy orders
profit = 0  # No profit until sold

# For sell orders (matched with previous buy)
buy_cost = buy_price × amount + buy_fee
sell_revenue = sell_price × amount - sell_fee
profit = sell_revenue - buy_cost
cumulative_profit += profit
```

---

## Algorithms

### Grid Level Spacing Algorithm

**Adaptive ATR-Based Spacing**:

```
Input: current_price, atr, num_levels, atr_multiplier
Output: List of price levels

1. base_step = atr × atr_multiplier
2. Enforce minimum spacing:
   min_step = current_price × 0.01  # 1%
   step = max(base_step, min_step)

3. Generate buy levels:
   level = current_price - step
   while level >= lower_bound and count < num_levels:
       add level to buy_levels
       level = level - step
       count++

4. Generate sell levels:
   level = current_price + step
   while level <= upper_bound and count < num_levels:
       add level to sell_levels
       level = level + step
       count++

5. Return sorted lists
```

**Complexity**: O(n) where n = num_levels

---

### Order Fill Detection Algorithm

**Polling-Based Detection**:

```
For each tracked order:
1. Query order status from exchange
2. If status == "closed":
   a. Calculate P&L
   b. Log to CSV
   c. Update equity
   d. Remove from tracked orders
3. If status == "cancelled":
   a. Log cancellation
   b. Remove from tracked orders
4. If status == "open":
   a. Keep tracking
```

**Optimization**: Batch status queries when possible

---

### Trend Detection Algorithm

**Multi-Indicator Confluence**:

```
Decision = ADX_condition AND RSI_condition AND EMA_condition

Where:
- ADX_condition = (ADX < threshold) → Ranging market
- RSI_condition = (lower_bound < RSI < upper_bound) → Neutral
- EMA_condition = (|price - EMA50| / EMA50 < max_distance) → Near EMA

Result:
- All true → ACTIVATE grid
- Any false → PAUSE grid
```

---

## Data Structures

### Order Tracking Dictionary

```python
placed_orders: Dict[str, Dict] = {
    "order_id_123": {
        "side": "buy",
        "price": 124.14,
        "amount": 0.025,
        "pair": "SOL_USDT",
        "timestamp": 1735463400.123
    }
}
```

### Grid Parameters Dictionary

```python
grid_params: Dict[str, Any] = {
    "buy_levels": [124.0, 123.5, 123.0],
    "sell_levels": [124.5, 125.0, 125.5],
    "base_amount": 0.025,
    "atr": 1.23,
    "current_price": 124.25
}
```

### Placement Result Dictionary

```python
result: Dict[str, Any] = {
    "buy_orders": ["id1", "id2"],
    "sell_orders": ["id3", "id4"],
    "failed_orders": [
        {
            "side": "sell",
            "price": 125.0,
            "amount": 0.025,
            "error": "BALANCE_NOT_ENOUGH"
        }
    ],
    "total_placed": 4,
    "total_failed": 1
}
```

---

## API Interfaces

### Gate.io REST API v4

**Base URLs**:

- Mainnet: `https://api.gateio.ws`
- Testnet: `https://fx-api-testnet.gateio.ws`

**Authentication Headers**:

```http
KEY: {api_key}
Timestamp: {unix_timestamp}
SIGN: {hmac_sha512_signature}
Content-Type: application/json
```

**Rate Limits**:

- Public endpoints: 900 requests/second
- Private endpoints: 900 requests/second
- Order placement: 100 requests/second

**Error Codes**:

- `BALANCE_NOT_ENOUGH`: Insufficient balance
- `INVALID_PARAM_VALUE`: Invalid parameter
- `ORDER_NOT_FOUND`: Order doesn't exist
- `INVALID_PRECISION`: Wrong decimal precision

---

## State Management

### State Lifecycle

```
1. Initialization
   ├─ Try load existing state
   ├─ If found: restore orders and parameters
   └─ If not: start fresh

2. During Operation
   ├─ Update state after each action
   ├─ Save state every iteration
   └─ Create periodic backups

3. On Shutdown
   ├─ Save final state
   ├─ Close all connections
   └─ Exit gracefully
```

### State Consistency

**Atomicity**:

- State saves are atomic (write to temp, then rename)
- Prevents corruption from crashes during write

**Validation**:

```python
def validate_state(state: Dict) -> bool:
    required_keys = ['version', 'timestamp', 'balances', 'active_orders']
    return all(key in state for key in required_keys)
```

---

## Error Handling

### Error Hierarchy

```
Exception
├── TradingError (base class)
│   ├── APIError
│   │   ├── RequestError
│   │   ├── AuthenticationError
│   │   └── RateLimitError
│   ├── ValidationError
│   │   ├── GridSpacingError
│   │   └── InsufficientBalanceError
│   └── StateError
│       ├── StateSaveError
│       └── StateLoadError
```

### Retry Logic

```python
max_retries = 3
retry_delay = 1.0  # seconds

for attempt in range(max_retries):
    try:
        result = api_call()
        return result
    except RequestError as e:
        if attempt < max_retries - 1:
            time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
            continue
        else:
            raise
```

### Critical Error Handling

**Max Drawdown Breach**:

1. Log critical message
2. Save state immediately
3. Cancel all open orders (optional)
4. Initiate graceful shutdown
5. Exit

**API Connection Loss**:

1. Log error
2. Retry with backoff
3. If persistent, pause trading
4. Continue monitoring
5. Resume when connection restored

---

## Performance Considerations

### Optimization Strategies

1. **API Call Batching**

   - Group balance queries
   - Batch order status checks
   - Reduce round-trips

2. **Caching**

   - Cache fee rates (rarely change)
   - Cache currency pair info
   - Cache indicator calculations when price unchanged

3. **Lazy Loading**

   - Only fetch candles when needed
   - Only check filled orders on active orders

4. **Memory Management**
   - Limit candle history (100 most recent)
   - Cleanup old backup files
   - Clear filled order tracking

### Typical Performance Metrics

- **Loop iteration time**: 1-3 seconds
- **API latency**: 100-300ms per request
- **State save time**: < 50ms
- **Memory usage**: ~50-100 MB
- **CPU usage**: < 5% average

---

## Extension Points

### Adding New Indicators

```python
# In market/indicators.py
def calculate_new_indicator(df: pd.DataFrame, period: int) -> float:
    """
    Calculate custom indicator.

    Args:
        df: OHLCV DataFrame
        period: Lookback period

    Returns:
        float: Indicator value
    """
    # Implementation
    pass

# In strategy/trend_filter.py
def can_activate_grid(..., new_threshold: float = 50.0):
    # Add to existing conditions
    new_value = calculate_new_indicator(df)
    new_condition = new_value < new_threshold

    return existing_conditions and new_condition
```

### Adding New Exchanges

```python
# Create connectors/new_exchange.py
class NewExchangeConnector:
    def __init__(self, mode: str):
        # Initialize
        pass

    def get_ticker_price(self, pair: str) -> float:
        # Implement
        pass

    def place_order(self, ...):
        # Implement
        pass

    # ... implement all required methods
```

### Adding New Risk Modules

```python
# Create risk/new_risk_module.py
class NewRiskProtection:
    def __init__(self, ...):
        pass

    def update(self, ...) -> bool:
        """
        Returns:
            bool: True if risk limit breached
        """
        pass

    def get_status(self) -> Dict:
        pass

# In main.py
self.new_risk = NewRiskProtection(...)
if self.new_risk.update(...):
    logger.critical("New risk limit breached")
    return False
```

---

## Testing Considerations

### Unit Test Coverage Areas

1. **Indicator Calculations**

   - Test with known datasets
   - Verify edge cases (insufficient data)

2. **Grid Generation**

   - Test different ATR values
   - Test boundary conditions
   - Test minimum spacing enforcement

3. **Order Validation**

   - Test insufficient balance
   - Test spacing too tight
   - Test spread too wide

4. **State Management**
   - Test save/load roundtrip
   - Test corruption recovery
   - Test missing files

### Integration Test Scenarios

1. **Full Trade Cycle**

   - Place buy order → Fill → Place sell order → Fill → P&L

2. **Drawdown Protection**

   - Simulate equity drop → Verify shutdown

3. **Trend Filter**

   - Various market conditions → Verify activation/pause

4. **Error Recovery**
   - Simulate API errors → Verify retry logic

### Testnet Testing Checklist

- [ ] Bot initializes correctly
- [ ] Orders placed successfully
- [ ] Orders fill and log to CSV
- [ ] State persists across restarts
- [ ] Drawdown protection triggers
- [ ] Trend filter pauses/resumes
- [ ] Insufficient balance handled gracefully
- [ ] Graceful shutdown saves state

---

## Security Considerations

### API Key Management

- Never log API keys
- Store in `.env` file (git-ignored)
- Use read-only keys when possible
- Enable IP whitelist on exchange

### Order Safety

- Validate all inputs before API calls
- Enforce minimum/maximum order sizes
- Check balance before every order
- Limit max open orders

### State Security

- Validate state structure before load
- Sanitize any user inputs
- Don't expose sensitive data in logs

---

## Configuration Parameters

### Environment Variables (.env)

```bash
# Required
GATE_IO_API_KEY=your_key
GATE_IO_API_SECRET=your_secret
GATE_IO_TESTNET_API_KEY=testnet_key
GATE_IO_TESTNET_API_SECRET=testnet_secret

# Optional
LOG_LEVEL=INFO
```

### Command Line Arguments

```bash
--mode {testnet|mainnet}    # Trading mode
--pair SOL_USDT             # Trading pair
--capital 50.0              # Initial capital
--interval 20               # Loop interval (seconds)
--max-drawdown 0.05         # Max drawdown (5%)
```

### Hardcoded Constants (can be parameterized)

```python
# In strategy/trend_filter.py
ADX_THRESHOLD = 28.0
RSI_LOWER_BOUND = 20.0
RSI_UPPER_BOUND = 80.0
MAX_EMA_DISTANCE_PCT = 5.0

# In strategy/grid_engine.py
ATR_MULTIPLIER = 0.5
GRID_RANGE_PCT = 0.10  # ±10%
MIN_SPACING_PCT = 0.01  # 1%

# In execution/order_manager.py
MAX_SPREAD_PCT = 0.3  # 0.3%
FEE_RATE = 0.001  # 0.1%

# In recovery/state_sync.py
MAX_BACKUPS = 10
BACKUP_DIR = "backups"
```

---

## Deployment

### System Requirements

- **OS**: Windows 10+, Linux, macOS
- **Python**: 3.8 or higher
- **RAM**: 256 MB minimum
- **Disk**: 100 MB for logs and backups
- **Network**: Stable internet connection

### Production Deployment Steps

1. **Prepare Environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure API Keys**

   - Copy `.env.example` to `.env`
   - Add mainnet API keys
   - Set proper permissions (`chmod 600 .env`)

3. **Test on Testnet First**

   ```bash
   python main.py --mode testnet --capital 30
   # Run for 24-48 hours
   ```

4. **Start Mainnet (Small)**

   ```bash
   python main.py --mode mainnet --capital 50
   ```

5. **Monitor**
   - Watch `grid_bot.log`
   - Check `trades.csv`
   - Verify on exchange UI

### Process Management

**Run in Background (Linux)**:

```bash
nohup python main.py --mode mainnet --capital 50 > output.log 2>&1 &
```

**As systemd Service** (Linux):

```ini
[Unit]
Description=Grid Scalping Bot
After=network.target

[Service]
Type=simple
User=trader
WorkingDirectory=/home/trader/grid-scalping
ExecStart=/home/trader/grid-scalping/.venv/bin/python main.py --mode mainnet --capital 50
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

---

## Appendix

### Glossary

- **Grid Trading**: Strategy placing buy/sell orders at fixed intervals
- **ATR**: Average True Range, volatility measure
- **ADX**: Average Directional Index, trend strength measure
- **RSI**: Relative Strength Index, momentum oscillator
- **EMA**: Exponential Moving Average
- **Drawdown**: Peak-to-trough decline in equity
- **Scalping**: High-frequency trading for small profits

### References

- Gate.io API Documentation: https://www.gate.io/docs/developers/apiv4/
- Technical Analysis Library: https://technical-analysis-library-in-python.readthedocs.io/
- Grid Trading Strategy: https://www.investopedia.com/terms/g/grid-trading.asp

### Version History

- **v1.0** (2025-12-29): Initial release
  - Core grid trading functionality
  - Multi-indicator trend filtering
  - Comprehensive risk management
  - State persistence and recovery

---

**End of Technical Specification**
