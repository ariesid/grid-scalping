"""
Trend Filter Module for Grid Trading Strategy

This module determines whether market conditions are suitable for grid trading.
It analyzes trend strength, momentum, and price positioning to decide if the
grid bot should be active or paused.

Key Concepts:
    - Grid trading works best in RANGING markets (low ADX)
    - Strong trends (high ADX) can cause one-sided grid fills
    - Extreme RSI values indicate overbought/oversold conditions
    - Price position relative to moving averages provides context

Requirements:
    - pandas: DataFrame handling
    - market.indicators: Technical analysis functions
"""

import logging
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np

# Import from our indicators module
from market.indicators import calculate_adx, calculate_ema

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    """
    Calculate Relative Strength Index (RSI).
    
    RSI measures momentum on a scale of 0-100:
        - Below 30: Oversold (potential buying opportunity)
        - 30-70: Normal range
        - Above 70: Overbought (potential selling opportunity)
    
    Args:
        prices: Close prices as pandas Series
        period: RSI period (default: 14)
        
    Returns:
        float: Latest RSI value (0-100)
        
    Raises:
        ValueError: If insufficient data or invalid period
        
    Example:
        >>> closes = pd.Series([100, 102, 101, 103, 105, 104, 106, 108])
        >>> rsi = calculate_rsi(closes, period=7)
        >>> print(f"RSI: {rsi:.2f}")
    """
    if period <= 0:
        raise ValueError(f"Period must be positive, got {period}")
    
    if len(prices) < period + 1:
        raise ValueError(
            f"Insufficient data: need at least {period + 1} prices, got {len(prices)}"
        )
    
    # Calculate price changes
    delta = prices.diff()
    
    # Separate gains and losses
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    
    # Calculate average gains and losses using Wilder's smoothing
    avg_gains = gains.ewm(alpha=1/period, adjust=False).mean()
    avg_losses = losses.ewm(alpha=1/period, adjust=False).mean()
    
    # Calculate RS (Relative Strength)
    rs = avg_gains / avg_losses
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    latest_rsi = float(rsi.iloc[-1])
    
    logger.debug(f"Calculated RSI({period}): {latest_rsi:.2f}")
    
    return latest_rsi


def is_ranging_market(adx_value: float, adx_threshold: float = 25.0) -> bool:
    """
    Determine if market is in a ranging (non-trending) state.
    
    Ranging markets are ideal for grid trading as prices oscillate
    within a range, allowing both buy and sell orders to fill.
    
    Args:
        adx_value: ADX value (0-100)
        adx_threshold: Maximum ADX for ranging market (default: 25)
        
    Returns:
        bool: True if ranging (ADX < threshold), False if trending
        
    Example:
        >>> if is_ranging_market(adx=20):
        ...     print("Ranging market - activate grid")
    """
    is_ranging = adx_value < adx_threshold
    
    if is_ranging:
        logger.info(f"Ranging market detected (ADX={adx_value:.2f} < {adx_threshold})")
    else:
        logger.info(f"Trending market detected (ADX={adx_value:.2f} >= {adx_threshold})")
    
    return is_ranging


def is_rsi_neutral(rsi_value: float, lower_bound: float = 30.0, upper_bound: float = 70.0) -> bool:
    """
    Check if RSI is in neutral range (not extreme).
    
    Extreme RSI values indicate potential reversals or exhaustion,
    which may not be ideal for entering new grid positions.
    
    Args:
        rsi_value: RSI value (0-100)
        lower_bound: Lower bound for neutral range (default: 30)
        upper_bound: Upper bound for neutral range (default: 70)
        
    Returns:
        bool: True if RSI is between bounds, False if extreme
        
    Example:
        >>> if is_rsi_neutral(rsi=50):
        ...     print("RSI neutral - safe to trade")
    """
    is_neutral = lower_bound <= rsi_value <= upper_bound
    
    if is_neutral:
        logger.debug(f"RSI neutral (RSI={rsi_value:.2f} in [{lower_bound}, {upper_bound}])")
    else:
        if rsi_value < lower_bound:
            logger.warning(f"RSI oversold (RSI={rsi_value:.2f} < {lower_bound})")
        else:
            logger.warning(f"RSI overbought (RSI={rsi_value:.2f} > {upper_bound})")
    
    return is_neutral


def get_price_position_vs_ema(current_price: float, ema_value: float) -> Dict[str, any]:
    """
    Analyze price position relative to EMA.
    
    This provides context on trend direction and potential support/resistance.
    
    Args:
        current_price: Current market price
        ema_value: EMA value (e.g., EMA(50))
        
    Returns:
        Dict with 'position' ('above'/'below'), 'distance' (absolute), 
        'distance_pct' (percentage)
        
    Example:
        >>> position = get_price_position_vs_ema(100.0, 98.0)
        >>> print(f"Price is {position['position']} EMA by {position['distance_pct']:.2f}%")
    """
    distance = current_price - ema_value
    distance_pct = (distance / ema_value) * 100
    position = "above" if distance > 0 else "below"
    
    result = {
        'position': position,
        'distance': abs(distance),
        'distance_pct': abs(distance_pct)
    }
    
    logger.debug(
        f"Price ${current_price:.2f} is {position} EMA ${ema_value:.2f} "
        f"by ${abs(distance):.2f} ({abs(distance_pct):.2f}%)"
    )
    
    return result


def analyze_market_conditions(
    df: pd.DataFrame,
    adx_threshold: float = 100.0,
    rsi_lower: float = 5.0,
    rsi_upper: float = 95.0
) -> Dict[str, any]:
    """
    Perform comprehensive market analysis using multiple indicators.
    
    Args:
        df: DataFrame with columns ['high', 'low', 'close'] (minimum required)
        
    Returns:
        Dict containing all calculated indicators and market assessment
        
    Raises:
        ValueError: If DataFrame is missing required columns or has insufficient data
        
    Example:
        >>> analysis = analyze_market_conditions(ohlcv_df)
        >>> print(f"ADX: {analysis['adx']:.2f}")
        >>> print(f"Market type: {analysis['market_type']}")
    """
    # Validate DataFrame
    required_columns = ['high', 'low', 'close']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(
            f"DataFrame missing required columns: {missing_columns}. "
            f"Required: {required_columns}"
        )
    
    if len(df) < 50:
        raise ValueError(
            f"Insufficient data: need at least 50 candles for analysis, got {len(df)}"
        )
    
    # Extract data
    highs = df['high']
    lows = df['low']
    closes = df['close']
    current_price = float(closes.iloc[-1])
    
    # Calculate indicators
    logger.info("Calculating market indicators...")
    
    try:
        adx = calculate_adx(highs, lows, closes, period=14)
    except Exception as e:
        logger.error(f"Failed to calculate ADX: {e}")
        adx = None
    
    try:
        ema_50 = calculate_ema(closes, period=50)
    except Exception as e:
        logger.error(f"Failed to calculate EMA(50): {e}")
        ema_50 = None
    
    try:
        rsi = calculate_rsi(closes, period=14)
    except Exception as e:
        logger.error(f"Failed to calculate RSI: {e}")
        rsi = None
    
    # Determine market type
    if adx is not None:
        market_type = "ranging" if adx < adx_threshold else "trending"
    else:
        market_type = "unknown"
    
    # Price position vs EMA
    price_vs_ema = None
    if ema_50 is not None:
        price_vs_ema = get_price_position_vs_ema(current_price, ema_50)
    
    # Compile analysis
    analysis = {
        'current_price': current_price,
        'adx': adx,
        'ema_50': ema_50,
        'rsi': rsi,
        'market_type': market_type,
        'is_ranging': adx < adx_threshold if adx is not None else None,
        'is_trending': adx >= adx_threshold if adx is not None else None,
        'rsi_neutral': is_rsi_neutral(rsi, rsi_lower, rsi_upper) if rsi is not None else None,
        'price_vs_ema': price_vs_ema,
        'candle_count': len(df)
    }
    
    adx_str = f"{adx:.2f}" if adx is not None else "N/A"
    rsi_str = f"{rsi:.2f}" if rsi is not None else "N/A"
    logger.info(
        f"Market analysis complete: {market_type.upper()} market "
        f"(ADX={adx_str}, RSI={rsi_str})"
    )
    
    return analysis


def can_activate_grid(
    df: pd.DataFrame,
    adx_threshold: float = 25.0,  # Ranging market: ADX below 25
    rsi_lower_bound: float = 30.0,  # Oversold threshold
    rsi_upper_bound: float = 70.0,  # Overbought threshold
    max_ema_distance_pct: float = 5.0  # Max 5% distance from EMA(50)
) -> bool:
    """
    Determine if grid trading should be activated based on market conditions.
    
    Grid trading is activated when:
        1. ADX < 25 (ranging market, not strong trend)
        2. RSI between 30-70 (not overbought/oversold)
        3. Price within 5% of EMA(50) (price not extended)
    
    Args:
        df: DataFrame with OHLCV data (columns: 'high', 'low', 'close' minimum)
        adx_threshold: Maximum ADX for grid activation (default: 25.0)
        rsi_lower_bound: Minimum RSI for neutral range (default: 30.0)
        rsi_upper_bound: Maximum RSI for neutral range (default: 70.0)
        max_ema_distance_pct: Max % distance from EMA(50) (default: 5.0)
        
    Returns:
        bool: True if grid should be activated, False if should pause
        
    Raises:
        ValueError: If DataFrame is invalid or has insufficient data
        
    Example:
        >>> import pandas as pd
        >>> # Load your OHLCV data
        >>> df = pd.DataFrame({
        ...     'high': [...],
        ...     'low': [...],
        ...     'close': [...]
        ... })
        >>> if can_activate_grid(df):
        ...     print("✓ Conditions favorable - activate grid bot")
        ... else:
        ...     print("✗ Unfavorable conditions - pause grid bot")
    """
    logger.info("=" * 60)
    logger.info("EVALUATING GRID ACTIVATION CONDITIONS")
    logger.info("=" * 60)
    
    try:
        # Perform market analysis
        analysis = analyze_market_conditions(
            df,
            adx_threshold=adx_threshold,
            rsi_lower=rsi_lower_bound,
            rsi_upper=rsi_upper_bound
        )
        
        # Check conditions
        conditions_met = []
        conditions_failed = []
        
        # Condition 1: Ranging market (ADX < threshold)
        if analysis['adx'] is not None:
            if analysis['is_ranging']:
                conditions_met.append(
                    f"✓ Ranging market (ADX={analysis['adx']:.2f} < {adx_threshold})"
                )
            else:
                conditions_failed.append(
                    f"✗ Strong trend detected (ADX={analysis['adx']:.2f} >= {adx_threshold})"
                )
        else:
            conditions_failed.append("✗ ADX calculation failed")
        
        # Condition 2: RSI in neutral range
        if analysis['rsi'] is not None:
            if analysis['rsi_neutral']:
                conditions_met.append(
                    f"✓ RSI neutral (RSI={analysis['rsi']:.2f} in [{rsi_lower_bound}, {rsi_upper_bound}])"
                )
            else:
                if analysis['rsi'] < rsi_lower_bound:
                    conditions_failed.append(
                        f"✗ RSI oversold (RSI={analysis['rsi']:.2f} < {rsi_lower_bound})"
                    )
                else:
                    conditions_failed.append(
                        f"✗ RSI overbought (RSI={analysis['rsi']:.2f} > {rsi_upper_bound})"
                    )
        else:
            conditions_failed.append("✗ RSI calculation failed")
        
        # Condition 3: Price not too far from EMA(50)
        if analysis['price_vs_ema'] is not None:
            distance_pct = analysis['price_vs_ema']['distance_pct']
            if distance_pct <= max_ema_distance_pct:
                conditions_met.append(
                    f"✓ Price near EMA(50) (distance={distance_pct:.2f}% <= {max_ema_distance_pct}%)"
                )
            else:
                conditions_failed.append(
                    f"✗ Price too far from EMA(50) (distance={distance_pct:.2f}% > {max_ema_distance_pct}%)"
                )
        else:
            conditions_failed.append("✗ EMA calculation failed")
        
        # Log results
        logger.info("\nConditions Met:")
        for condition in conditions_met:
            logger.info(f"  {condition}")
        
        if conditions_failed:
            logger.info("\nConditions Failed:")
            for condition in conditions_failed:
                logger.info(f"  {condition}")
        
        # Decision
        can_activate = len(conditions_failed) == 0
        
        logger.info("=" * 60)
        if can_activate:
            logger.info("DECISION: ✓ ACTIVATE GRID TRADING")
            logger.info("Market conditions are favorable for grid strategy")
        else:
            logger.info("DECISION: ✗ PAUSE GRID TRADING")
            logger.info("Market conditions not suitable - wait for better setup")
        logger.info("=" * 60)
        
        return can_activate
    
    except Exception as e:
        logger.error(f"Error evaluating grid activation: {e}")
        logger.info("DECISION: ✗ PAUSE GRID (Error occurred)")
        return False


def get_market_regime_description(df: pd.DataFrame) -> str:
    """
    Get a human-readable description of current market regime.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        str: Description of market conditions and trading recommendations
        
    Example:
        >>> description = get_market_regime_description(df)
        >>> print(description)
    """
    try:
        analysis = analyze_market_conditions(df)
        
        lines = []
        lines.append("=" * 60)
        lines.append("MARKET REGIME ANALYSIS")
        lines.append("=" * 60)
        lines.append(f"\nCurrent Price: ${analysis['current_price']:.2f}")
        lines.append(f"Candles Analyzed: {analysis['candle_count']}")
        lines.append("")
        
        # Trend Analysis
        lines.append("TREND ANALYSIS:")
        if analysis['adx'] is not None:
            adx = analysis['adx']
            if adx < 20:
                trend_strength = "Very Weak"
                recommendation = "Excellent for grid trading"
            elif adx < 25:
                trend_strength = "Weak"
                recommendation = "Good for grid trading"
            elif adx < 40:
                trend_strength = "Moderate"
                recommendation = "Use caution with grid trading"
            elif adx < 60:
                trend_strength = "Strong"
                recommendation = "Consider trend-following instead"
            else:
                trend_strength = "Very Strong"
                recommendation = "Avoid grid trading"
            
            lines.append(f"  ADX: {adx:.2f} - {trend_strength} Trend")
            lines.append(f"  Market Type: {analysis['market_type'].upper()}")
            lines.append(f"  Recommendation: {recommendation}")
        else:
            lines.append("  ADX: Not available")
        
        lines.append("")
        
        # Momentum Analysis
        lines.append("MOMENTUM ANALYSIS:")
        if analysis['rsi'] is not None:
            rsi = analysis['rsi']
            if rsi < 30:
                momentum = "Oversold"
                signal = "Potential buying opportunity"
            elif rsi > 70:
                momentum = "Overbought"
                signal = "Potential selling opportunity"
            else:
                momentum = "Neutral"
                signal = "Balanced conditions"
            
            lines.append(f"  RSI(14): {rsi:.2f} - {momentum}")
            lines.append(f"  Signal: {signal}")
        else:
            lines.append("  RSI: Not available")
        
        lines.append("")
        
        # Price Position
        lines.append("PRICE POSITION:")
        if analysis['ema_50'] is not None and analysis['price_vs_ema'] is not None:
            ema = analysis['ema_50']
            pos = analysis['price_vs_ema']
            
            lines.append(f"  EMA(50): ${ema:.2f}")
            lines.append(
                f"  Price is {pos['position']} EMA by "
                f"${pos['distance']:.2f} ({pos['distance_pct']:.2f}%)"
            )
            
            if pos['position'] == 'above':
                lines.append(f"  Bias: Bullish (price above EMA)")
            else:
                lines.append(f"  Bias: Bearish (price below EMA)")
        else:
            lines.append("  EMA: Not available")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    except Exception as e:
        return f"Error generating market regime description: {e}"


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage with mock SOL/USDT OHLCV data.
    """
    
    print("\n" + "=" * 70)
    print("TREND FILTER - Example Usage with Mock Data")
    print("=" * 70)
    print()
    
    # Generate mock OHLCV data (100 candles)
    np.random.seed(42)
    num_candles = 100
    
    base_price = 100.0
    
    # Scenario 1: Ranging market (ideal for grid trading)
    print("-" * 70)
    print("SCENARIO 1: RANGING MARKET")
    print("-" * 70)
    print()
    
    closes_ranging = []
    highs_ranging = []
    lows_ranging = []
    
    for i in range(num_candles):
        # Oscillating price with no clear trend
        oscillation = 5 * np.sin(i / 10)
        noise = np.random.normal(0, 1)
        close = base_price + oscillation + noise
        high = close + abs(np.random.normal(0.5, 0.3))
        low = close - abs(np.random.normal(0.5, 0.3))
        
        closes_ranging.append(close)
        highs_ranging.append(high)
        lows_ranging.append(low)
    
    df_ranging = pd.DataFrame({
        'high': highs_ranging,
        'low': lows_ranging,
        'close': closes_ranging
    })
    
    print("Generated 100 candles of ranging market data")
    print(f"Price range: ${min(lows_ranging):.2f} - ${max(highs_ranging):.2f}")
    print()
    
    # Test grid activation for ranging market
    can_activate_ranging = can_activate_grid(df_ranging)
    print()
    
    # Show detailed analysis
    print(get_market_regime_description(df_ranging))
    print()
    
    # Scenario 2: Trending market (not ideal for grid trading)
    print("-" * 70)
    print("SCENARIO 2: TRENDING MARKET")
    print("-" * 70)
    print()
    
    closes_trending = []
    highs_trending = []
    lows_trending = []
    
    for i in range(num_candles):
        # Strong upward trend
        trend = i * 0.5
        noise = np.random.normal(0, 0.5)
        close = base_price + trend + noise
        high = close + abs(np.random.normal(0.5, 0.3))
        low = close - abs(np.random.normal(0.5, 0.3))
        
        closes_trending.append(close)
        highs_trending.append(high)
        lows_trending.append(low)
    
    df_trending = pd.DataFrame({
        'high': highs_trending,
        'low': lows_trending,
        'close': closes_trending
    })
    
    print("Generated 100 candles of trending market data")
    print(f"Price range: ${min(lows_trending):.2f} - ${max(highs_trending):.2f}")
    print()
    
    # Test grid activation for trending market
    can_activate_trending = can_activate_grid(df_trending)
    print()
    
    # Show detailed analysis
    print(get_market_regime_description(df_trending))
    print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Ranging Market:
  - Grid Activation: {'✓ YES' if can_activate_ranging else '✗ NO'}
  - Ideal for grid trading due to price oscillation

Trending Market:
  - Grid Activation: {'✓ YES' if can_activate_trending else '✗ NO'}
  - Not ideal for grid trading due to directional movement

Grid Activation Criteria:
  1. ADX < 25 (ranging market)
  2. RSI between 30-70 (not extreme)
  3. Price within 5% of EMA(50) (reasonable entry)

All criteria must be met for grid activation.
""")
    print("=" * 70)
    print()
