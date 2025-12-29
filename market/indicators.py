"""
Technical Indicators Module for SOL/USDT Grid Scalping Bot

This module provides real-time trend and volatility detection indicators
for making informed trading decisions. Supports both pandas Series and list inputs.

Key Indicators:
    - ADX (Average Directional Index): Measures trend strength
    - ATR (Average True Range): Measures volatility
    - EMA (Exponential Moving Average): Smoothed price trend
    
Requirements:
    - ta: Technical Analysis Library (recommended)
    - pandas: Data manipulation
    - numpy: Numerical operations
"""

import logging
from typing import Union, List, Optional
import numpy as np
import pandas as pd

# Try importing ta library, fall back to manual calculations if unavailable
try:
    import ta
    from ta.trend import ADXIndicator
    from ta.volatility import AverageTrueRange
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    logging.warning(
        "ta library not available. Using manual calculations. "
        "Install with: pip install ta"
    )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _to_series(data: Union[List[float], pd.Series, np.ndarray]) -> pd.Series:
    """
    Convert input data to pandas Series.
    
    Args:
        data: Input data as list, numpy array, or pandas Series
        
    Returns:
        pd.Series: Data as pandas Series
    """
    if isinstance(data, pd.Series):
        return data
    elif isinstance(data, (list, np.ndarray)):
        return pd.Series(data)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


def calculate_ema(prices: Union[List[float], pd.Series], period: int = 14) -> float:
    """
    Calculate Exponential Moving Average (EMA).
    
    EMA gives more weight to recent prices, making it more responsive to 
    price changes than Simple Moving Average (SMA).
    
    Args:
        prices: Price data (close prices) as list or pandas Series
        period: EMA period (default: 14)
        
    Returns:
        float: Latest EMA value
        
    Raises:
        ValueError: If insufficient data or invalid period
        
    Example:
        >>> prices = [100, 102, 101, 103, 105, 104, 106]
        >>> ema = calculate_ema(prices, period=5)
        >>> print(f"EMA: {ema:.2f}")
    """
    if period <= 0:
        raise ValueError(f"Period must be positive, got {period}")
    
    prices_series = _to_series(prices)
    
    if len(prices_series) < period:
        raise ValueError(
            f"Insufficient data: need at least {period} prices, got {len(prices_series)}"
        )
    
    # Calculate EMA using pandas ewm (exponentially weighted moving average)
    ema_series = prices_series.ewm(span=period, adjust=False).mean()
    latest_ema = float(ema_series.iloc[-1])
    
    logger.debug(f"Calculated EMA({period}): {latest_ema:.4f}")
    return latest_ema


def calculate_atr(
    high: Union[List[float], pd.Series],
    low: Union[List[float], pd.Series],
    close: Union[List[float], pd.Series],
    period: int = 14
) -> float:
    """
    Calculate Average True Range (ATR) for volatility measurement.
    
    ATR measures market volatility by decomposing the entire range of an asset
    price for a given period. Higher ATR indicates higher volatility.
    
    Args:
        high: High prices as list or pandas Series
        low: Low prices as list or pandas Series
        close: Close prices as list or pandas Series
        period: ATR period (default: 14)
        
    Returns:
        float: Latest ATR value
        
    Raises:
        ValueError: If insufficient data, mismatched lengths, or invalid period
        
    Example:
        >>> highs = [105, 107, 106, 108, 110]
        >>> lows = [100, 102, 101, 103, 105]
        >>> closes = [102, 105, 103, 106, 108]
        >>> atr = calculate_atr(highs, lows, closes, period=5)
        >>> print(f"ATR: {atr:.2f}")
    """
    if period <= 0:
        raise ValueError(f"Period must be positive, got {period}")
    
    high_series = _to_series(high)
    low_series = _to_series(low)
    close_series = _to_series(close)
    
    # Validate data lengths
    if not (len(high_series) == len(low_series) == len(close_series)):
        raise ValueError(
            f"Data length mismatch: high={len(high_series)}, "
            f"low={len(low_series)}, close={len(close_series)}"
        )
    
    if len(close_series) < period + 1:
        raise ValueError(
            f"Insufficient data: need at least {period + 1} candles, "
            f"got {len(close_series)}"
        )
    
    if TA_AVAILABLE:
        # Use ta library for ATR calculation
        atr_indicator = AverageTrueRange(
            high=high_series,
            low=low_series,
            close=close_series,
            window=period
        )
        atr_value = float(atr_indicator.average_true_range().iloc[-1])
    else:
        # Manual ATR calculation
        # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
        prev_close = close_series.shift(1)
        
        tr1 = high_series - low_series
        tr2 = abs(high_series - prev_close)
        tr3 = abs(low_series - prev_close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR is the moving average of True Range
        atr_series = true_range.rolling(window=period).mean()
        atr_value = float(atr_series.iloc[-1])
    
    logger.debug(f"Calculated ATR({period}): {atr_value:.4f}")
    return atr_value


def calculate_adx(
    high: Union[List[float], pd.Series],
    low: Union[List[float], pd.Series],
    close: Union[List[float], pd.Series],
    period: int = 14
) -> float:
    """
    Calculate Average Directional Index (ADX) for trend strength measurement.
    
    ADX quantifies trend strength on a scale of 0-100:
        - 0-25: Absent or weak trend
        - 25-50: Strong trend
        - 50-75: Very strong trend
        - 75-100: Extremely strong trend
    
    Args:
        high: High prices as list or pandas Series
        low: Low prices as list or pandas Series
        close: Close prices as list or pandas Series
        period: ADX period (default: 14)
        
    Returns:
        float: Latest ADX value (0-100)
        
    Raises:
        ValueError: If insufficient data, mismatched lengths, or invalid period
        
    Example:
        >>> highs = [105, 107, 106, 108, 110, 112, 115, 114, 116, 118]
        >>> lows = [100, 102, 101, 103, 105, 107, 110, 109, 111, 113]
        >>> closes = [102, 105, 103, 106, 108, 110, 113, 112, 114, 116]
        >>> adx = calculate_adx(highs, lows, closes, period=5)
        >>> print(f"ADX: {adx:.2f}")
    """
    if period <= 0:
        raise ValueError(f"Period must be positive, got {period}")
    
    high_series = _to_series(high)
    low_series = _to_series(low)
    close_series = _to_series(close)
    
    # Validate data lengths
    if not (len(high_series) == len(low_series) == len(close_series)):
        raise ValueError(
            f"Data length mismatch: high={len(high_series)}, "
            f"low={len(low_series)}, close={len(close_series)}"
        )
    
    # ADX requires more data for accurate calculation
    min_required = period * 2 + 1
    if len(close_series) < min_required:
        raise ValueError(
            f"Insufficient data: need at least {min_required} candles for ADX({period}), "
            f"got {len(close_series)}"
        )
    
    if TA_AVAILABLE:
        # Use ta library for ADX calculation
        adx_indicator = ADXIndicator(
            high=high_series,
            low=low_series,
            close=close_series,
            window=period
        )
        adx_value = float(adx_indicator.adx().iloc[-1])
    else:
        # Manual ADX calculation
        # Step 1: Calculate +DM and -DM (Directional Movement)
        high_diff = high_series.diff()
        low_diff = -low_series.diff()
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        # Step 2: Calculate True Range (TR)
        prev_close = close_series.shift(1)
        tr1 = high_series - low_series
        tr2 = abs(high_series - prev_close)
        tr3 = abs(low_series - prev_close)
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Step 3: Smooth the indicators using Wilder's smoothing
        atr = true_range.ewm(alpha=1/period, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean() / atr
        
        # Step 4: Calculate DX (Directional Index)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        
        # Step 5: Calculate ADX (smoothed DX)
        adx_series = dx.ewm(alpha=1/period, adjust=False).mean()
        adx_value = float(adx_series.iloc[-1])
    
    # Ensure ADX is within valid range
    adx_value = max(0.0, min(100.0, adx_value))
    
    logger.debug(f"Calculated ADX({period}): {adx_value:.2f}")
    return adx_value


def is_strong_trend(adx_value: float, threshold: float = 25.0) -> bool:
    """
    Determine if there is a strong trend based on ADX value.
    
    A strong trend indicates that the market is moving directionally,
    which is favorable for trend-following strategies but may require
    caution for grid trading strategies.
    
    Args:
        adx_value: ADX value (0-100)
        threshold: Minimum ADX value to consider trend as strong (default: 25)
        
    Returns:
        bool: True if trend is strong (ADX >= threshold), False otherwise
        
    Example:
        >>> adx = calculate_adx(highs, lows, closes)
        >>> if is_strong_trend(adx):
        ...     print("Strong trend detected - adjust grid strategy")
        ... else:
        ...     print("Weak trend - ideal for grid trading")
    """
    if not 0 <= adx_value <= 100:
        logger.warning(f"ADX value {adx_value} outside normal range [0, 100]")
    
    is_strong = adx_value >= threshold
    
    if is_strong:
        logger.info(f"Strong trend detected (ADX={adx_value:.2f} >= {threshold})")
    else:
        logger.info(f"Weak trend detected (ADX={adx_value:.2f} < {threshold})")
    
    return is_strong


def calculate_volatility_ratio(current_atr: float, historical_atr: float) -> float:
    """
    Calculate volatility ratio compared to historical average.
    
    This helps determine if current volatility is higher or lower than usual,
    which can inform grid spacing decisions.
    
    Args:
        current_atr: Current ATR value
        historical_atr: Historical average ATR value
        
    Returns:
        float: Volatility ratio (1.0 = normal, >1.0 = higher volatility)
        
    Example:
        >>> current = calculate_atr(recent_highs, recent_lows, recent_closes)
        >>> historical = calculate_atr(all_highs, all_lows, all_closes, period=100)
        >>> ratio = calculate_volatility_ratio(current, historical)
        >>> print(f"Volatility is {ratio:.2f}x normal")
    """
    if historical_atr <= 0:
        raise ValueError(f"Historical ATR must be positive, got {historical_atr}")
    
    ratio = current_atr / historical_atr
    logger.debug(f"Volatility ratio: {ratio:.2f}x (current={current_atr:.4f}, historical={historical_atr:.4f})")
    
    return ratio


def get_market_regime(
    adx_value: float,
    atr_value: float,
    adx_threshold: float = 25.0,
    atr_threshold: float = None
) -> str:
    """
    Determine current market regime based on trend strength and volatility.
    
    Market regimes:
        - 'trending_high_volatility': Strong trend + high volatility
        - 'trending_low_volatility': Strong trend + low volatility
        - 'ranging_high_volatility': Weak trend + high volatility (ideal for grid)
        - 'ranging_low_volatility': Weak trend + low volatility
    
    Args:
        adx_value: ADX value indicating trend strength
        atr_value: ATR value indicating volatility
        adx_threshold: Threshold for strong trend (default: 25)
        atr_threshold: Threshold for high volatility (optional, default: median)
        
    Returns:
        str: Market regime classification
        
    Example:
        >>> adx = calculate_adx(highs, lows, closes)
        >>> atr = calculate_atr(highs, lows, closes)
        >>> regime = get_market_regime(adx, atr)
        >>> print(f"Market regime: {regime}")
    """
    is_trending = adx_value >= adx_threshold
    
    # If no ATR threshold provided, use a simple heuristic
    if atr_threshold is None:
        is_high_volatility = True  # Default to high volatility
    else:
        is_high_volatility = atr_value >= atr_threshold
    
    if is_trending and is_high_volatility:
        regime = 'trending_high_volatility'
    elif is_trending and not is_high_volatility:
        regime = 'trending_low_volatility'
    elif not is_trending and is_high_volatility:
        regime = 'ranging_high_volatility'
    else:
        regime = 'ranging_low_volatility'
    
    logger.info(
        f"Market regime: {regime} (ADX={adx_value:.2f}, ATR={atr_value:.4f})"
    )
    
    return regime


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage with mock SOL/USDT data.
    This demonstrates all indicator calculations with realistic price movements.
    """
    
    print("=" * 70)
    print("Technical Indicators - Example Usage with Mock SOL/USDT Data")
    print("=" * 70)
    print()
    
    # Mock SOL/USDT price data (50 candles)
    # Simulating a trending market with some volatility
    np.random.seed(42)  # For reproducibility
    
    base_price = 100.0
    num_candles = 50
    
    # Generate realistic OHLC data with trend and volatility
    closes = []
    highs = []
    lows = []
    
    for i in range(num_candles):
        # Add trend component (upward)
        trend = i * 0.5
        
        # Add random volatility
        volatility = np.random.normal(0, 2)
        
        close = base_price + trend + volatility
        high = close + abs(np.random.normal(1, 0.5))
        low = close - abs(np.random.normal(1, 0.5))
        
        closes.append(close)
        highs.append(high)
        lows.append(low)
    
    print(f"Generated {num_candles} candles of mock SOL/USDT data")
    print(f"Price range: ${min(lows):.2f} - ${max(highs):.2f}")
    print(f"Latest close: ${closes[-1]:.2f}")
    print()
    
    # Calculate EMA
    print("-" * 70)
    print("1. Exponential Moving Average (EMA)")
    print("-" * 70)
    
    ema_9 = calculate_ema(closes, period=9)
    ema_21 = calculate_ema(closes, period=21)
    
    print(f"EMA(9):  ${ema_9:.2f}")
    print(f"EMA(21): ${ema_21:.2f}")
    print(f"Current price vs EMA(9): {((closes[-1] / ema_9 - 1) * 100):+.2f}%")
    print()
    
    # Calculate ATR
    print("-" * 70)
    print("2. Average True Range (ATR) - Volatility Measurement")
    print("-" * 70)
    
    atr_14 = calculate_atr(highs, lows, closes, period=14)
    atr_percentage = (atr_14 / closes[-1]) * 100
    
    print(f"ATR(14): ${atr_14:.2f}")
    print(f"ATR as % of price: {atr_percentage:.2f}%")
    print(f"Interpretation: Average price movement per candle")
    print()
    
    # Calculate ADX
    print("-" * 70)
    print("3. Average Directional Index (ADX) - Trend Strength")
    print("-" * 70)
    
    adx_14 = calculate_adx(highs, lows, closes, period=14)
    
    print(f"ADX(14): {adx_14:.2f}")
    
    if adx_14 < 25:
        strength = "Weak/Absent"
        strategy = "Good for grid trading"
    elif adx_14 < 50:
        strength = "Strong"
        strategy = "Use caution with grid trading"
    elif adx_14 < 75:
        strength = "Very Strong"
        strategy = "Favor trend-following strategies"
    else:
        strength = "Extremely Strong"
        strategy = "Strong trend - avoid grid trading"
    
    print(f"Trend strength: {strength}")
    print(f"Trading strategy: {strategy}")
    print()
    
    # Check if strong trend
    print("-" * 70)
    print("4. Trend Analysis")
    print("-" * 70)
    
    strong_trend = is_strong_trend(adx_14, threshold=25)
    print(f"Strong trend detected: {strong_trend}")
    print()
    
    # Calculate volatility ratio
    print("-" * 70)
    print("5. Volatility Analysis")
    print("-" * 70)
    
    # Use different periods for comparison
    atr_short = calculate_atr(highs[-20:], lows[-20:], closes[-20:], period=7)
    atr_long = calculate_atr(highs, lows, closes, period=30)
    
    vol_ratio = calculate_volatility_ratio(atr_short, atr_long)
    
    print(f"Recent ATR(7):     ${atr_short:.2f}")
    print(f"Long-term ATR(30): ${atr_long:.2f}")
    print(f"Volatility ratio:  {vol_ratio:.2f}x")
    
    if vol_ratio > 1.2:
        print("Interpretation: Higher than normal volatility - widen grid spacing")
    elif vol_ratio < 0.8:
        print("Interpretation: Lower than normal volatility - tighten grid spacing")
    else:
        print("Interpretation: Normal volatility - use standard grid spacing")
    print()
    
    # Market regime classification
    print("-" * 70)
    print("6. Market Regime Classification")
    print("-" * 70)
    
    regime = get_market_regime(adx_14, atr_14, adx_threshold=25)
    
    print(f"Current market regime: {regime.replace('_', ' ').title()}")
    
    regime_strategies = {
        'trending_high_volatility': 'Reduce grid density, widen spacing, reduce position sizes',
        'trending_low_volatility': 'Tight stop-loss, small grid positions',
        'ranging_high_volatility': 'IDEAL for grid trading! Use wider spacing',
        'ranging_low_volatility': 'Grid trading with tighter spacing'
    }
    
    print(f"Recommended strategy: {regime_strategies.get(regime, 'Monitor closely')}")
    print()
    
    # Trading signal summary
    print("=" * 70)
    print("TRADING SIGNAL SUMMARY")
    print("=" * 70)
    print(f"Current Price:     ${closes[-1]:.2f}")
    print(f"EMA(9):            ${ema_9:.2f}")
    print(f"EMA(21):           ${ema_21:.2f}")
    print(f"ATR(14):           ${atr_14:.2f} ({atr_percentage:.2f}% of price)")
    print(f"ADX(14):           {adx_14:.2f} ({strength} trend)")
    print(f"Market Regime:     {regime.replace('_', ' ').title()}")
    print()
    
    # Grid trading recommendation
    if not strong_trend and vol_ratio > 0.8:
        grid_spacing = atr_14 * 0.5  # Half of ATR as grid spacing
        print("✓ CONDITIONS FAVORABLE FOR GRID TRADING")
        print(f"  Suggested grid spacing: ${grid_spacing:.2f} ({(grid_spacing/closes[-1]*100):.2f}%)")
    else:
        print("⚠ CONDITIONS NOT IDEAL FOR GRID TRADING")
        print("  Consider trend-following strategy or wait for ranging market")
    
    print()
    print("=" * 70)
