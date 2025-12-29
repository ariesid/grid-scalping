# Grid Scalping Bot - Configuration Guide

## Overview

This document explains the key configuration parameters that control when and how the grid trading bot operates. These settings ensure the bot only trades in favorable market conditions and maintains profitable grid spacing.

## Current Production Setup

**Environment:** Gate.io Mainnet  
**Trading Pair:** SOL/USDT  
**Initial Capital:** $50 USDT  
**Mode:** Live Trading (Real Money)  
**Status:** Active

### Account Balances

- **USDT:** $59.61 available
- **SOL:** 0.001152 available (~$0.14)
- **Total Equity:** ~$59.75

### Order Configuration

- **Order Size:** Dynamic (calculated to meet $5 minimum)
  - At SOL price $124: ~0.044 SOL per order (~$5.50)
- **Fee Rate:** 0.1% (Gate.io standard)
- **Min Grid Spacing:** 0.4% (2x fees for profit)
- **Loop Interval:** 20 seconds

### Risk Management

- **Max Drawdown:** 5% ($2.50 loss limit from peak)
- **Drawdown Protection:** ✅ Enabled
- **Equity Calculation:** Includes available + locked funds + SOL value
- **Emergency Stop:** Automatic on drawdown breach

### Inventory Management

- **Strategy:** Start with USDT, accumulate SOL through buy orders
- **Buy-First Mode:** ✅ Active (skips sell orders until SOL acquired)
- **Target Balance:** 50% USDT / 50% SOL (achieved over time)

---

## Trend Filter Configuration

Located in: `strategy/trend_filter.py` → `can_activate_grid()`

These parameters determine whether market conditions are suitable for grid trading.

### ADX Threshold

```python
adx_threshold: float = 25.0
```

**What it is:** Average Directional Index measures trend strength

- **0-25**: Weak trend (ranging/sideways market) ✅ **GRID FRIENDLY**
- **25-50**: Developing trend
- **50-75**: Strong trend ⚠️ **RISKY FOR GRID**
- **75-100**: Very strong trend ❌ **AVOID GRID**

**Current Setting:** `25.0` → Updated to `28.0` for production

- Bot activates grid when **ADX < 28** (ranging to mildly trending markets)
- Slightly relaxed from testnet (25.0) to allow more trading opportunities
- Higher ADX indicates directional movement that can cause one-sided grid fills
- Grid trading works best in consolidation, not strong trends

**Why this matters:**
In a strong uptrend (high ADX), all your BUY orders fill but SELL orders don't → you're holding lots of assets at high prices. In a strong downtrend, all your SELL orders fill but BUY orders don't → you're holding cash while missing the recovery.

---

### RSI Bounds

```python
rsi_lower_bound: float = 30.0   # Oversold threshold
rsi_upper_bound: float = 70.0   # Overbought threshold
```

**What it is:** Relative Strength Index measures momentum/overbought/oversold

- **0-30**: Oversold (potential reversal down)
- **30-70**: Neutral range ✅ **GRID FRIENDLY**
- **70-100**: Overbought (potential reversal down)

**Current Settings:**

- Lower: `20.0` - Bot pauses when **RSI < 20** (deeply oversold)
- Upper: `80.0` - Bot pauses when **RSI > 80** (deeply overbought)

**Production Note:** Widened from testnet (30/70) to (20/80) to allow more trading opportunities in mainnet ranging markets.

**Why this matters:**

- **RSI > 70**: Price is near resistance, likely to reverse down → placing grid now risks all BUY orders filling on the way down
- **RSI < 30**: Price is near support, likely to reverse up → placing grid now risks all SELL orders filling on the way up
- **RSI 30-70**: Balanced momentum, no extreme exhaustion → safe for grid

**Example from your trading:**
When you ran the bot, RSI was 93 (very overbought, near resistance). With old test settings (RSI upper = 95), it placed orders. With new settings (RSI upper = 70), it would have **paused** and waited for RSI to cool down below 70.

---

### EMA Distance Limit

```python
max_ema_distance_pct: float = 5.0
```

**What it is:** Maximum allowed distance between current price and 50-period EMA

- EMA(50) acts as dynamic support/resistance and trend indicator
- Price should stay relatively close to EMA(50) in ranging markets

**Current Setting:** `5.0%`

- Bot activates grid when price is within **±5% of EMA(50)**
- If price is >5% above or below EMA, conditions are too extended

**Why this matters:**
When price strays too far from EMA(50), it often indicates:

- Strong directional move (trend, not range)
- Potential mean reversion coming (price snaps back to EMA)
- Risk of whipsaw if you place grid during extended move

Keeping price near EMA ensures you're trading in consolidation zones.

---

## Order Manager Configuration

Located in: `execution/order_manager.py` → `validate_grid_spacing()`

These parameters ensure grid orders are spaced profitably.

### Grid Spacing Validation

```python
min_spacing_pct = 2.0 * 2 * fee_rate
```

**What it is:** Minimum percentage distance between consecutive grid levels

**Calculation:**

- `fee_rate`: Trading fee (e.g., 0.2% = 0.002)
- Multiplier: `2.0` (profit margin multiplier)
- Total fees: `2 * fee_rate` (entry fee + exit fee)
- **Formula:** `min_spacing = 2.0 × 2 × fee_rate`

**Example with 0.1% fee (Gate.io mainnet):**

```
min_spacing = 2.0 × 2 × 0.001 = 0.004 = 0.4%
```

**What this means:**
Each grid level must be at least **0.4% apart** to ensure:

1. Entry fee covered: 0.1%
2. Exit fee covered: 0.1%
3. Profit margin: 0.2% (2x the fees)

**Production Note:** Gate.io mainnet has 0.1% fees (better than testnet 0.2%), allowing tighter grid spacing and more frequent trades.

**Why this matters:**
If grid levels are too close (e.g., 0.1% apart), you'll break even or lose money:

- Buy at $124 → Pay 0.1% fee → Cost: $124.12
- Sell at $124.12 → Pay 0.1% fee → Net: $124.00
- **Result:** $0 profit (fees ate it all)

With 0.4% spacing (current setup):

- Buy at $124 → Pay 0.1% fee → Cost: $124.12
- Sell at $124.50 → Pay 0.1% fee → Net: $124.38
- **Result:** $0.26 profit (0.2% gain after fees on $124 order)

**Real-world example from production ($50 capital):**

- Order size: 0.044 SOL (~$5.50)
- Profit per cycle: ~$0.05-$0.15
- Target: 5-10 trades/day = $0.50-$1.50/day (1-3% daily return)

---

### Profit Multiplier

```python
profit_multiplier = 2.0x
```

**What it is:** How much larger grid spacing should be compared to total fees

**Current Setting:** `2.0x`

- Grid spacing = 2x the total trading fees
- Ensures meaningful profit above break-even

**Why 2.0x?**

- **1.0x**: Break-even (covers fees only, no profit)
- **2.0x**: Covers fees + 100% profit margin ✅ **BALANCED**
- **3.0x**: Covers fees + 200% profit margin (safer but fewer fills)

**Trade-off:**

- Lower multiplier (1.5x) → More grid fills, smaller profit per trade
- Higher multiplier (3.0x) → Fewer grid fills, larger profit per trade

`2.0x` balances frequency and profitability.

---

## Complete Trading Rules (Production Mainnet)

The bot will **ACTIVATE GRID** only when ALL conditions are met:

✅ **ADX < 28** → Ranging to mildly trending market
✅ **RSI 20-80** → Not deeply oversold or overbought
✅ **Price within ±5% of EMA(50)** → Price not extended
✅ **Grid spacing ≥ 0.4%** → Profitable after 0.1% fees
✅ **Sufficient inventory** → Has USDT for buys or SOL for sells

The bot will **PAUSE GRID** if ANY condition fails:

❌ ADX ≥ 28 → Strong trend detected
❌ RSI < 20 → Deeply oversold (near support)
❌ RSI > 80 → Deeply overbought (near resistance)
❌ Price > 5% from EMA(50) → Price extended
❌ Grid spacing < 0.4% → Unprofitable spacing
❌ Insufficient balance → Skips orders (buys or sells) until balance restored

**Production Note:** Bot automatically adapts to imbalanced inventory. If you have only USDT (no SOL), it will place buy orders first and skip sells until SOL is acquired.

---

## Adjusting Configuration

### For More Aggressive Trading

If you want the bot to trade more frequently (accept more risk):

```python
# trend_filter.py
adx_threshold = 30.0          # Accept slightly trending markets
rsi_lower_bound = 25.0        # Accept more oversold conditions
rsi_upper_bound = 75.0        # Accept more overbought conditions
max_ema_distance_pct = 8.0    # Accept more extended price moves
```

### For More Conservative Trading

If you want stricter conditions (safer but fewer trades):

```python
# trend_filter.py
adx_threshold = 20.0          # Only very calm markets
rsi_lower_bound = 35.0        # Narrower RSI range
rsi_upper_bound = 65.0        # Narrower RSI range
max_ema_distance_pct = 3.0    # Price must stay very close to EMA

# order_manager.py
profit_multiplier = 3.0       # Wider grid spacing for more profit per trade
```

### For Higher Fee Exchanges

If your exchange has higher fees (e.g., 0.5%):

The grid spacing automatically adjusts based on `fee_rate`:

```python
# With 0.5% fee:
min_spacing = 2.0 × 2 × 0.005 = 0.02 = 2.0%
```

No code changes needed - the formula handles it automatically.

---

## Testing Your Configuration

Before running with real money:

1. **Backtest** on historical data to verify profitability
2. **Paper trade** on testnet for 1-2 weeks
3. **Monitor metrics**:

   - Win rate (should be >60%)
   - Average profit per trade (should be >0.4% with 0.2% fee)
   - Max drawdown (should stay <5%)
   - Grid fill frequency (should have balanced buy/sell fills)

4. **Start small** on mainnet with $100-500 capital

---

## Configuration Summary Table

| Parameter              | Testnet Value | Production Value | What It Controls             | Impact                                       |
| ---------------------- | ------------- | ---------------- | ---------------------------- | -------------------------------------------- |
| `adx_threshold`        | 25.0          | **28.0**         | Maximum trend strength       | Higher = more trades, slightly riskier       |
| `rsi_lower_bound`      | 30.0          | **20.0**         | Oversold threshold           | Lower = trade in more oversold conditions    |
| `rsi_upper_bound`      | 70.0          | **80.0**         | Overbought threshold         | Higher = trade in more overbought conditions |
| `max_ema_distance_pct` | 5.0%          | **5.0%**         | Max price deviation from EMA | Higher = trade when price more extended      |
| `profit_multiplier`    | 2.0x          | **2.0x**         | Grid spacing vs fees         | Higher = more profit/trade, fewer fills      |
| `fee_rate`             | 0.2%          | **0.1%**         | Trading fees                 | Lower = tighter grids possible               |
| `min_grid_spacing`     | 0.8%          | **0.4%**         | Min distance between levels  | Tighter = more frequent fills                |
| `order_size`           | 0.025 SOL     | **~0.044 SOL**   | Amount per order             | Calculated to meet $5 minimum                |
| `capital`              | $30           | **$50**          | Total trading capital        | Higher = more/larger orders                  |

**Production Changes Summary:**

- Relaxed ADX (25→28) and RSI (30-70→20-80) for more trading opportunities
- Lower fees (0.2%→0.1%) allow tighter grid spacing (0.8%→0.4%)
- Dynamic order sizing ensures Gate.io $5 minimum is met
- Added inventory management to handle imbalanced USDT/SOL

---

## See Also

- **Environment Variables**: See `.env` for API keys and trading mode
- **Command Line Arguments**: See `main.py --help` for runtime parameters
- **Trading Logs**: Check `grid_bot.log` for detailed decision logs
- **Trade History**: See `trades.csv` for filled order records

---

## Questions?

**Q: Why is my bot not placing orders?**
A: Check `grid_bot.log` for the exact condition that failed (ADX, RSI, or EMA distance).

**Q: How do I know if my grid spacing is profitable?**
A: Check logs for "Grid spacing validated" message showing minimum spacing met.

**Q: The bot paused during a good market. Why?**
A: It likely detected early warning signs (ADX rising, RSI extending) before they became obvious.

**Q: Can I use different settings for different pairs?**
A: Not currently - config is global. Consider running separate bot instances for different strategies.

---

**Last Updated:** December 29, 2025  
**Version:** 1.1 (Production Mainnet)  
**Environment:** Gate.io Mainnet with $50 USDT capital
