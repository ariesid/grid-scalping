# SOL/USDT Grid Scalping Bot

An intelligent automated grid trading bot for SOL/USDT on Gate.io exchange. Features adaptive grid levels, trend filtering, comprehensive risk management, and state persistence.

## 🎯 Features

### Core Trading

- **Adaptive Grid Trading**: Dynamic grid levels based on ATR (Average True Range)
- **Trend Filtering**: Only trades in favorable market conditions (ADX, RSI, EMA analysis)
- **Smart Order Placement**: Automatic spacing optimization to ensure profitability after fees
- **Balance-Aware**: Automatically skips orders when insufficient inventory

### Risk Management

- **Drawdown Protection**: Automatic shutdown when max drawdown is breached (default: 5%)
- **Trailing Stops**: Optional profit protection
- **Spread Monitoring**: Skips trading when order book spread is too wide
- **Fee-Aware Validation**: Ensures grid spacing covers trading fees

### Reliability

- **State Persistence**: Recovers gracefully from crashes/restarts
- **Automatic Backups**: Regular state backups with cleanup
- **Error Recovery**: Handles API errors and partial order failures
- **Graceful Shutdown**: Saves state on Ctrl+C

### Monitoring

- **Comprehensive Logging**: Detailed logs to file and console
- **Trade CSV Export**: All trades logged to `trades.csv`
- **Real-time Statistics**: P&L, drawdown, equity tracking

## 📋 Requirements

- Python 3.8 or higher
- Gate.io account (testnet or mainnet)
- API keys with spot trading permissions

## 🚀 Installation

1. **Clone or download the repository**

2. **Create virtual environment**

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows
# or
source .venv/bin/activate  # Linux/Mac
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Configure API keys**
   Copy `.env.example` to `.env` and add your API keys:

```bash
# Testnet (for testing)
GATE_IO_TESTNET_API_KEY=your_testnet_key
GATE_IO_TESTNET_API_SECRET=your_testnet_secret

# Mainnet (for live trading)
GATE_IO_API_KEY=your_mainnet_key
GATE_IO_API_SECRET=your_mainnet_secret
```

## ⚙️ Configuration

### Key Parameters

Edit [CONFIG.md](CONFIG.md) for detailed parameter explanations.

**Trend Filter** (`strategy/trend_filter.py`):

- `adx_threshold`: 28.0 (lower = more ranging markets)
- `rsi_bounds`: [20, 80] (avoid extremes)
- `max_ema_distance_pct`: 5.0% (stay near EMA50)

**Risk Management**:

- `max_drawdown`: 5% (default)
- `initial_capital`: Set to your actual capital
- `loop_interval`: 20 seconds (check frequency)

## 🎮 Usage

### Testnet (Recommended for Testing)

```bash
# Start with $30 testnet capital
py main.py --mode testnet --capital 30

# Custom settings
py main.py --mode testnet --capital 100 --interval 30 --max-drawdown 0.10
```

### Mainnet (Live Trading)

⚠️ **WARNING: Real money at risk!**

```bash
# Start small!
py main.py --mode mainnet --capital 50

# With custom settings
py main.py --mode mainnet --capital 100 --interval 20 --max-drawdown 0.05
```

### Command Line Arguments

| Argument         | Default    | Description                          |
| ---------------- | ---------- | ------------------------------------ |
| `--mode`         | `testnet`  | Trading mode: `testnet` or `mainnet` |
| `--pair`         | `SOL_USDT` | Trading pair                         |
| `--capital`      | `10000`    | Initial capital amount               |
| `--interval`     | `20`       | Loop interval in seconds             |
| `--max-drawdown` | `0.05`     | Max allowed drawdown (5%)            |

## 📊 How It Works

### Strategy Overview

The bot implements a **grid scalping strategy** with intelligent market filtering:

1. **Market Analysis**

   - Calculates ADX (trend strength), RSI (momentum), EMA(50)
   - Only activates grid in ranging markets (low ADX, neutral RSI, near EMA)

2. **Grid Generation**

   - Creates buy levels below current price
   - Creates sell levels above current price
   - Spacing based on ATR for volatility adaptation

3. **Order Placement**

   - Checks balances (skips if insufficient inventory)
   - Validates grid spacing vs fees
   - Places orders on exchange

4. **Monitoring**

   - Checks for filled orders every iteration
   - Updates P&L and equity
   - Monitors drawdown

5. **Risk Protection**
   - Stops trading if drawdown exceeds limit
   - Pauses grid in unfavorable market conditions
   - Saves state continuously

### Example Trade Flow

```
Price: $124.00
├─ Buy Grid
│  ├─ Buy @ $123.50 (0.025 SOL)
│  └─ Buy @ $123.00 (0.025 SOL)
└─ Sell Grid
   ├─ Sell @ $124.50 (0.025 SOL)
   └─ Sell @ $125.00 (0.025 SOL)

When price drops to $123.50 → Buy fills → You have SOL
When price rises to $124.50 → Sell fills → Profit!
```

## 🛡️ Safety Recommendations

### Before Going Live

1. **Test on Testnet First**

   - Run for 24-48 hours minimum
   - Verify behavior in different market conditions
   - Check logs for any errors

2. **Start Small**

   - Use capital you can afford to lose
   - Start with $50-$100 maximum
   - Increase only after proven stable

3. **Check Prerequisites**

   ```bash
   # Verify your balance matches capital setting
   py check_balance.py
   ```

4. **Monitor Closely**
   - Watch logs during first 24 hours
   - Check `trades.csv` for activity
   - Verify orders on exchange UI

### Inventory Management

Grid trading requires **balanced inventory**:

- **Need both USDT and SOL** to trade both sides
- If you only have USDT: Bot will buy first, then can sell
- If you only have SOL: Bot will sell first, then can buy

**Recommended Split** (for $100 capital):

- $50 USDT (for buy orders)
- $50 worth of SOL (for sell orders)

## 📁 Project Structure

```
grid-scalping/
├── main.py                 # Main bot orchestrator
├── requirements.txt        # Python dependencies
├── .env                    # API keys (create from .env.example)
├── bot_state.json         # Persistent state
├── trades.csv             # Trade log
├── grid_bot.log           # Detailed logs
├── CONFIG.md              # Parameter documentation
├── README.md              # This file
├── connectors/
│   └── gate_io.py         # Gate.io API wrapper
├── strategy/
│   ├── grid_engine.py     # Grid level generation
│   └── trend_filter.py    # Market condition analysis
├── execution/
│   └── order_manager.py   # Order placement logic
├── risk/
│   ├── drawdown_monitor.py # Drawdown protection
│   └── trailing_stop.py    # Trailing stop logic
├── market/
│   └── indicators.py       # Technical indicators
├── recovery/
│   └── state_sync.py       # State persistence
├── utils/
│   └── trade_logger.py     # CSV trade logging
└── backups/
    └── bot_state_backup_*.json
```

## 🔍 Monitoring

### Check Current Status

```bash
# View recent activity
Get-Content grid_bot.log -Tail 50

# Check trades
Get-Content trades.csv

# Check balances
py check_balance.py
```

### Important Log Messages

- `✓ ACTIVATE GRID TRADING` - Bot is trading
- `✗ PAUSE GRID TRADING` - Market conditions unfavorable
- `🛑 MAX DRAWDOWN BREACHED` - Risk limit hit, bot stopped
- `⚠️ Insufficient SOL/USDT` - Need more inventory

## ⚠️ Troubleshooting

### Bot Shuts Down Immediately

**Cause**: Max drawdown triggered
**Fix**:

- Check equity calculation is correct
- Increase `--max-drawdown` if needed
- Ensure capital parameter matches your balance

### "BALANCE_NOT_ENOUGH" Error

**Cause**: Insufficient inventory for orders
**Fix**:

- Check balance: `py check_balance.py`
- Buy SOL or USDT manually to balance inventory
- Wait for buy orders to fill (bot will auto-balance over time)

### No Orders Placed

**Cause**: Unfavorable market conditions
**Check logs**: Look for "PAUSE GRID TRADING"
**Reason**: High ADX (trending), extreme RSI, or price far from EMA
**Action**: Wait for conditions to improve (bot will auto-resume)

### Order Size Too Small

**Cause**: Gate.io minimum is $5 per order
**Fix**: Bot now auto-calculates order size to meet minimum
**If still issues**: Increase capital parameter

## 📈 Performance Expectations

### Testnet Results

- **Profit per trade**: ~$0.025 (2.5 cents)
- **5 successful trades**: $0.12 profit
- **All orders filled correctly**: ✅

### Realistic Mainnet Expectations

With $50 capital:

- **Order size**: ~$5.50 per order
- **Profit per cycle**: ~$0.05-$0.15 (depends on price movement)
- **Daily target**: 5-10 trades = $0.50-$1.50/day (1-3% daily)
- **Risk**: Max 5% drawdown = $2.50 loss limit

**Note**: Grid trading works best in ranging markets. During strong trends, profits may be lower.

## 🔐 Security

- Never share your `.env` file
- Use read-only API keys if possible
- Enable IP whitelist on Gate.io
- Start with API keys that have withdrawal disabled
- Monitor account regularly

## 📝 License

This project is for educational purposes. Use at your own risk.

## ⚡ Quick Start Checklist

- [ ] Tested on testnet for 24+ hours
- [ ] API keys configured in `.env`
- [ ] Balance matches `--capital` parameter
- [ ] Inventory balanced (50% USDT, 50% SOL)
- [ ] Started with small capital ($50-$100)
- [ ] Monitoring logs and trades regularly
- [ ] Understand max drawdown protection
- [ ] Know how to stop (Ctrl+C)

## 📞 Support

Check logs first: `grid_bot.log`
Review config: [CONFIG.md](CONFIG.md)
Read code comments for detailed explanations

---

**Disclaimer**: Trading cryptocurrencies involves substantial risk of loss. This bot is provided as-is with no guarantees. Always test thoroughly on testnet before using real money. Never invest more than you can afford to lose.
