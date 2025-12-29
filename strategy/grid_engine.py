"""
Adaptive Grid Engine for SOL/USDT Scalping Bot

This module generates dynamic grid trading levels based on volatility (ATR),
price range, and market trend bias. The grid adapts to market conditions for
optimal order placement.

Key Features:
    - ATR-based adaptive spacing
    - Trend-aware asymmetric grids
    - Configurable range and level density
    - Boundary enforcement
"""

import logging
from typing import Dict, List, Tuple
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GridEngine:
    """
    Adaptive grid level generator for cryptocurrency trading.
    
    This engine creates buy and sell price levels that adapt to market
    volatility and trend direction, optimizing grid placement for
    different market conditions.
    """
    
    def __init__(self, atr_multiplier: float = 0.5):
        """
        Initialize grid engine.
        
        Args:
            atr_multiplier: Multiplier for ATR-based spacing (default: 0.5)
        """
        self.atr_multiplier = atr_multiplier
        logger.info(f"GridEngine initialized with ATR multiplier: {atr_multiplier}")
    
    def _calculate_bounds(
        self, 
        current_price: float, 
        grid_range_pct: float
    ) -> Tuple[float, float]:
        """
        Calculate upper and lower price bounds for grid.
        
        Args:
            current_price: Current market price
            grid_range_pct: Grid range as percentage (e.g., 0.10 for 10%)
            
        Returns:
            Tuple[float, float]: (lower_bound, upper_bound)
        """
        half_range = grid_range_pct / 2
        lower_bound = current_price * (1 - half_range)
        upper_bound = current_price * (1 + half_range)
        
        logger.debug(
            f"Price bounds: [{lower_bound:.2f}, {upper_bound:.2f}] "
            f"(±{grid_range_pct*100:.1f}% from ${current_price:.2f})"
        )
        
        return lower_bound, upper_bound
    
    def _calculate_level_distribution(
        self, 
        total_levels: int, 
        trend_bias: str
    ) -> Tuple[int, int]:
        """
        Calculate buy and sell level counts based on trend bias.
        
        Args:
            total_levels: Total number of grid levels
            trend_bias: Trend direction - "neutral", "bullish", or "bearish"
            
        Returns:
            Tuple[int, int]: (buy_levels_count, sell_levels_count)
            
        Raises:
            ValueError: If trend_bias is invalid
        """
        if trend_bias == "neutral":
            buy_count = total_levels // 2
            sell_count = total_levels - buy_count
        elif trend_bias == "bullish":
            # Bullish: 30% buy, 70% sell (favor selling into strength)
            buy_count = int(total_levels * 0.3)
            sell_count = total_levels - buy_count
        elif trend_bias == "bearish":
            # Bearish: 70% buy, 30% sell (favor buying the dip)
            buy_count = int(total_levels * 0.7)
            sell_count = total_levels - buy_count
        else:
            raise ValueError(
                f"Invalid trend_bias: '{trend_bias}'. "
                f"Must be 'neutral', 'bullish', or 'bearish'"
            )
        
        logger.info(
            f"Level distribution for {trend_bias} bias: "
            f"{buy_count} buy levels, {sell_count} sell levels"
        )
        
        return buy_count, sell_count
    
    def _generate_levels_atr_based(
        self,
        start_price: float,
        count: int,
        step: float,
        direction: str,
        lower_bound: float,
        upper_bound: float,
        min_spacing_pct: float = 0.01  # Minimum 1% spacing between levels
    ) -> List[float]:
        """
        Generate price levels using ATR-based spacing with minimum percentage enforcement.
        
        Args:
            start_price: Starting price for level generation
            count: Number of levels to generate
            step: Price step size (ATR-based)
            direction: "up" or "down"
            lower_bound: Minimum allowed price
            upper_bound: Maximum allowed price
            min_spacing_pct: Minimum spacing as decimal (e.g., 0.01 for 1%)
            
        Returns:
            List[float]: List of price levels within bounds
        """
        levels = []
        current = start_price
        
        multiplier = 1 if direction == "up" else -1
        
        for i in range(count):
            # Move by ATR step or minimum percentage, whichever is larger
            atr_step = step * multiplier
            min_pct_step = current * min_spacing_pct * multiplier
            
            # Use the larger of ATR step or minimum percentage step
            actual_step = atr_step if abs(atr_step) > abs(min_pct_step) else min_pct_step
            current = current + actual_step
            
            # Enforce bounds
            if current < lower_bound:
                current = lower_bound
            elif current > upper_bound:
                current = upper_bound
            
            # Add level if within bounds and meets minimum spacing
            if lower_bound <= current <= upper_bound:
                if not levels:
                    levels.append(current)
                else:
                    # Check if spacing from last level meets minimum
                    spacing_pct = abs(current - levels[-1]) / levels[-1]
                    if spacing_pct >= min_spacing_pct * 0.8:  # 80% of minimum to allow small tolerance
                        levels.append(current)
        
        return levels
    
    def generate_grid_levels(
        self,
        current_price: float,
        atr_value: float,
        grid_range_pct: float = 0.10,
        total_levels: int = 20,
        trend_bias: str = "neutral"
    ) -> Dict[str, List[float]]:
        """
        Generate adaptive grid levels for trading.
        
        This is the main function that creates buy and sell price levels
        based on current market conditions, volatility, and trend direction.
        
        Args:
            current_price: Current market price (e.g., 100.0)
            atr_value: Average True Range value (e.g., 2.5)
            grid_range_pct: Grid range as decimal (e.g., 0.10 for ±10%). Default: 0.10
            total_levels: Total number of grid levels. Default: 20
            trend_bias: Market trend - "neutral", "bullish", or "bearish". Default: "neutral"
            
        Returns:
            Dict with 'buy_levels' and 'sell_levels' as sorted lists of floats
            
        Raises:
            ValueError: If parameters are invalid
            
        Example:
            >>> engine = GridEngine(atr_multiplier=0.5)
            >>> grid = engine.generate_grid_levels(
            ...     current_price=100.0,
            ...     atr_value=2.5,
            ...     grid_range_pct=0.10,
            ...     total_levels=20,
            ...     trend_bias="bullish"
            ... )
            >>> print(f"Buy levels: {len(grid['buy_levels'])}")
            >>> print(f"Sell levels: {len(grid['sell_levels'])}")
        """
        # Validate inputs
        if current_price <= 0:
            raise ValueError(f"current_price must be positive, got {current_price}")
        if atr_value <= 0:
            raise ValueError(f"atr_value must be positive, got {atr_value}")
        if grid_range_pct <= 0 or grid_range_pct >= 1:
            raise ValueError(f"grid_range_pct must be between 0 and 1, got {grid_range_pct}")
        if total_levels < 2:
            raise ValueError(f"total_levels must be at least 2, got {total_levels}")
        
        logger.info(
            f"Generating grid: price=${current_price:.2f}, ATR=${atr_value:.2f}, "
            f"range={grid_range_pct*100:.1f}%, levels={total_levels}, bias={trend_bias}"
        )
        
        # Calculate price bounds
        lower_bound, upper_bound = self._calculate_bounds(current_price, grid_range_pct)
        
        # Calculate ATR-based step size
        step = self.atr_multiplier * atr_value
        logger.debug(f"ATR-based step size: ${step:.4f}")
        
        # Determine level distribution
        buy_count, sell_count = self._calculate_level_distribution(total_levels, trend_bias)
        
        # Generate buy levels (below current price)
        buy_levels = self._generate_levels_atr_based(
            start_price=current_price,
            count=buy_count,
            step=step,
            direction="down",
            lower_bound=lower_bound,
            upper_bound=current_price
        )
        
        # Generate sell levels (above current price)
        sell_levels = self._generate_levels_atr_based(
            start_price=current_price,
            count=sell_count,
            step=step,
            direction="up",
            lower_bound=current_price,
            upper_bound=upper_bound
        )
        
        # Sort levels
        buy_levels = sorted(buy_levels, reverse=True)  # Highest to lowest
        sell_levels = sorted(sell_levels)  # Lowest to highest
        
        # Log summary
        logger.info(
            f"Grid generated: {len(buy_levels)} buy levels "
            f"[${min(buy_levels):.2f} - ${max(buy_levels):.2f}], "
            f"{len(sell_levels)} sell levels "
            f"[${min(sell_levels):.2f} - ${max(sell_levels):.2f}]"
        )
        
        return {
            "buy_levels": buy_levels,
            "sell_levels": sell_levels
        }


def generate_grid_levels(
    current_price: float,
    atr_value: float,
    grid_range_pct: float = 0.10,
    total_levels: int = 20,
    trend_bias: str = "neutral",
    atr_multiplier: float = 0.5
) -> Dict[str, List[float]]:
    """
    Convenience function to generate grid levels without instantiating GridEngine.
    
    Args:
        current_price: Current market price
        atr_value: Average True Range value
        grid_range_pct: Grid range as decimal (default: 0.10 for ±10%)
        total_levels: Total number of grid levels (default: 20)
        trend_bias: Market trend - "neutral", "bullish", or "bearish" (default: "neutral")
        atr_multiplier: ATR spacing multiplier (default: 0.5)
        
    Returns:
        Dict with 'buy_levels' and 'sell_levels' as lists of floats
        
    Example:
        >>> from strategy.grid_engine import generate_grid_levels
        >>> grid = generate_grid_levels(100.0, 2.5, 0.10, 20, "neutral")
        >>> print(grid['buy_levels'][:3])  # First 3 buy levels
    """
    engine = GridEngine(atr_multiplier=atr_multiplier)
    return engine.generate_grid_levels(
        current_price=current_price,
        atr_value=atr_value,
        grid_range_pct=grid_range_pct,
        total_levels=total_levels,
        trend_bias=trend_bias
    )


def calculate_grid_spacing_stats(grid: Dict[str, List[float]]) -> Dict[str, float]:
    """
    Calculate statistics about grid level spacing.
    
    Args:
        grid: Grid dictionary with 'buy_levels' and 'sell_levels'
        
    Returns:
        Dict with spacing statistics
        
    Example:
        >>> grid = generate_grid_levels(100.0, 2.5, 0.10, 20, "neutral")
        >>> stats = calculate_grid_spacing_stats(grid)
        >>> print(f"Average buy spacing: ${stats['avg_buy_spacing']:.2f}")
    """
    buy_levels = grid['buy_levels']
    sell_levels = grid['sell_levels']
    
    stats = {}
    
    # Calculate buy level spacing
    if len(buy_levels) > 1:
        buy_spacings = [abs(buy_levels[i] - buy_levels[i+1]) 
                       for i in range(len(buy_levels)-1)]
        stats['avg_buy_spacing'] = np.mean(buy_spacings)
        stats['min_buy_spacing'] = np.min(buy_spacings)
        stats['max_buy_spacing'] = np.max(buy_spacings)
    
    # Calculate sell level spacing
    if len(sell_levels) > 1:
        sell_spacings = [abs(sell_levels[i] - sell_levels[i+1]) 
                        for i in range(len(sell_levels)-1)]
        stats['avg_sell_spacing'] = np.mean(sell_spacings)
        stats['min_sell_spacing'] = np.min(sell_spacings)
        stats['max_sell_spacing'] = np.max(sell_spacings)
    
    # Overall stats
    if buy_levels and sell_levels:
        stats['price_range'] = max(sell_levels) - min(buy_levels)
        stats['mid_gap'] = sell_levels[0] - buy_levels[0]
    
    return stats


def visualize_grid(
    grid: Dict[str, List[float]], 
    current_price: float,
    max_display: int = 10
) -> str:
    """
    Create a text-based visualization of the grid levels.
    
    Args:
        grid: Grid dictionary with 'buy_levels' and 'sell_levels'
        current_price: Current market price
        max_display: Maximum levels to display per side (default: 10)
        
    Returns:
        str: Formatted grid visualization
        
    Example:
        >>> grid = generate_grid_levels(100.0, 2.5, 0.10, 20, "bullish")
        >>> print(visualize_grid(grid, 100.0))
    """
    buy_levels = grid['buy_levels'][:max_display]
    sell_levels = grid['sell_levels'][:max_display]
    
    lines = []
    lines.append("=" * 60)
    lines.append("GRID LEVELS VISUALIZATION")
    lines.append("=" * 60)
    lines.append("")
    
    # Sell levels (top to bottom)
    lines.append(f"{'SELL LEVELS':<20} {'Price':>12} {'Distance':>15}")
    lines.append("-" * 60)
    for i, price in enumerate(reversed(sell_levels), 1):
        distance = price - current_price
        distance_pct = (distance / current_price) * 100
        lines.append(
            f"  Sell #{i:<13} ${price:>10.2f}  "
            f"(+${distance:.2f} / +{distance_pct:.2f}%)"
        )
    
    # Current price marker
    lines.append("")
    lines.append(f"{'>>> CURRENT PRICE':<20} ${current_price:>10.2f}  {'<<<':>15}")
    lines.append("")
    
    # Buy levels (top to bottom)
    lines.append(f"{'BUY LEVELS':<20} {'Price':>12} {'Distance':>15}")
    lines.append("-" * 60)
    for i, price in enumerate(buy_levels, 1):
        distance = current_price - price
        distance_pct = (distance / current_price) * 100
        lines.append(
            f"  Buy #{i:<14} ${price:>10.2f}  "
            f"(-${distance:.2f} / -{distance_pct:.2f}%)"
        )
    
    lines.append("")
    lines.append("=" * 60)
    
    return "\n".join(lines)


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage demonstrating grid generation for different market conditions.
    """
    
    print("\n" + "=" * 70)
    print("ADAPTIVE GRID ENGINE - Example Usage")
    print("=" * 70)
    print()
    
    # Example parameters (SOL/USDT)
    current_price = 100.0
    atr_value = 2.5
    grid_range_pct = 0.10  # ±10% range
    total_levels = 20
    
    print(f"Market Conditions:")
    print(f"  Current Price: ${current_price:.2f}")
    print(f"  ATR (14): ${atr_value:.2f}")
    print(f"  Grid Range: ±{grid_range_pct*100:.1f}%")
    print(f"  Total Levels: {total_levels}")
    print()
    
    # Example 1: Neutral market (50/50 distribution)
    print("\n" + "-" * 70)
    print("EXAMPLE 1: NEUTRAL MARKET (50/50 distribution)")
    print("-" * 70)
    
    grid_neutral = generate_grid_levels(
        current_price=current_price,
        atr_value=atr_value,
        grid_range_pct=grid_range_pct,
        total_levels=total_levels,
        trend_bias="neutral",
        atr_multiplier=0.5
    )
    
    print(f"\nGenerated {len(grid_neutral['buy_levels'])} buy levels:")
    print(f"  Range: ${min(grid_neutral['buy_levels']):.2f} - ${max(grid_neutral['buy_levels']):.2f}")
    print(f"  Top 5: {[f'${x:.2f}' for x in grid_neutral['buy_levels'][:5]]}")
    
    print(f"\nGenerated {len(grid_neutral['sell_levels'])} sell levels:")
    print(f"  Range: ${min(grid_neutral['sell_levels']):.2f} - ${max(grid_neutral['sell_levels']):.2f}")
    print(f"  Bottom 5: {[f'${x:.2f}' for x in grid_neutral['sell_levels'][:5]]}")
    
    stats_neutral = calculate_grid_spacing_stats(grid_neutral)
    print(f"\nSpacing Statistics:")
    print(f"  Avg buy spacing: ${stats_neutral.get('avg_buy_spacing', 0):.2f}")
    print(f"  Avg sell spacing: ${stats_neutral.get('avg_sell_spacing', 0):.2f}")
    print(f"  Mid gap: ${stats_neutral.get('mid_gap', 0):.2f}")
    
    # Example 2: Bullish market (30/70 distribution)
    print("\n" + "-" * 70)
    print("EXAMPLE 2: BULLISH MARKET (30% buy, 70% sell)")
    print("-" * 70)
    
    grid_bullish = generate_grid_levels(
        current_price=current_price,
        atr_value=atr_value,
        grid_range_pct=grid_range_pct,
        total_levels=total_levels,
        trend_bias="bullish",
        atr_multiplier=0.5
    )
    
    print(f"\nGenerated {len(grid_bullish['buy_levels'])} buy levels (30%)")
    print(f"  Range: ${min(grid_bullish['buy_levels']):.2f} - ${max(grid_bullish['buy_levels']):.2f}")
    
    print(f"\nGenerated {len(grid_bullish['sell_levels'])} sell levels (70%)")
    print(f"  Range: ${min(grid_bullish['sell_levels']):.2f} - ${max(grid_bullish['sell_levels']):.2f}")
    
    print(f"\nStrategy: Favor selling into strength, limited downside buying")
    
    # Example 3: Bearish market (70/30 distribution)
    print("\n" + "-" * 70)
    print("EXAMPLE 3: BEARISH MARKET (70% buy, 30% sell)")
    print("-" * 70)
    
    grid_bearish = generate_grid_levels(
        current_price=current_price,
        atr_value=atr_value,
        grid_range_pct=grid_range_pct,
        total_levels=total_levels,
        trend_bias="bearish",
        atr_multiplier=0.5
    )
    
    print(f"\nGenerated {len(grid_bearish['buy_levels'])} buy levels (70%)")
    print(f"  Range: ${min(grid_bearish['buy_levels']):.2f} - ${max(grid_bearish['buy_levels']):.2f}")
    
    print(f"\nGenerated {len(grid_bearish['sell_levels'])} sell levels (30%)")
    print(f"  Range: ${min(grid_bearish['sell_levels']):.2f} - ${max(grid_bearish['sell_levels']):.2f}")
    
    print(f"\nStrategy: Accumulate on dips, limited upside selling")
    
    # Visualization
    print("\n" + "-" * 70)
    print("GRID VISUALIZATION (Neutral Market)")
    print("-" * 70)
    print(visualize_grid(grid_neutral, current_price, max_display=8))
    
    # Example 4: High volatility scenario (larger ATR multiplier)
    print("\n" + "-" * 70)
    print("EXAMPLE 4: HIGH VOLATILITY (ATR multiplier = 1.0)")
    print("-" * 70)
    
    grid_high_vol = generate_grid_levels(
        current_price=current_price,
        atr_value=atr_value,
        grid_range_pct=grid_range_pct,
        total_levels=total_levels,
        trend_bias="neutral",
        atr_multiplier=1.0  # Wider spacing for high volatility
    )
    
    stats_high_vol = calculate_grid_spacing_stats(grid_high_vol)
    print(f"\nWith wider ATR multiplier (1.0 vs 0.5):")
    print(f"  Avg buy spacing: ${stats_high_vol.get('avg_buy_spacing', 0):.2f}")
    print(f"  Avg sell spacing: ${stats_high_vol.get('avg_sell_spacing', 0):.2f}")
    print(f"  (Compare to normal: ${stats_neutral.get('avg_buy_spacing', 0):.2f})")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Trend Bias Impact:
  • Neutral:  50/50 distribution - balanced grid for ranging markets
  • Bullish: 30/70 distribution - more sell orders to capture upside
  • Bearish: 70/30 distribution - more buy orders to accumulate dips

ATR-Based Spacing:
  • Adapts to volatility automatically
  • Default multiplier: 0.5 (conservative)
  • Increase for high volatility, decrease for low volatility
  • Current step size: ${0.5 * atr_value:.2f} (0.5 × ${atr_value:.2f})

Grid Range:
  • ±{grid_range_pct*100:.1f}% from current price
  • Lower bound: ${current_price * (1 - grid_range_pct/2):.2f}
  • Upper bound: ${current_price * (1 + grid_range_pct/2):.2f}
""")
    
    print("=" * 70)
    print()
