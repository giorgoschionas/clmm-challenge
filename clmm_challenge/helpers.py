import numpy as np
import math

# ============================================================================
# Uniswap V3 Concentrated Liquidity Functions
# ============================================================================

def price_to_tick(price: float) -> int:
    """Convert price to tick index"""
    return int(np.floor(np.log(price) / np.log(1.0001)))




def get_position_value_vec(L, external_p_current, sqrt_p_current, sqrt_p_lower, sqrt_p_upper):
    """
    Vectorized mark-to-market value of a Uniswap v3 position in terms of quote asset (token1).

    Handles arrays where each element may be above, below, or in range independently.

    Args:
        L: Liquidity amount, array-like shape (num_trajectories,)
        external_p_current: Current price of the asset (not sqrt), array-like shape (num_trajectories,)
        sqrt_p_current: Current sqrt(price), array-like shape (num_trajectories,)
        sqrt_p_lower: Lower bound sqrt(price), array-like shape (num_trajectories,)
        sqrt_p_upper: Upper bound sqrt(price), array-like shape (num_trajectories,)

    Returns:
        np.ndarray: Total value in terms of token1, shape (num_trajectories,)
    """
    L = np.atleast_1d(np.asarray(L, dtype=np.float64))
    sqrt_p_current = np.atleast_1d(np.asarray(sqrt_p_current, dtype=np.float64))
    sqrt_p_lower = np.atleast_1d(np.asarray(sqrt_p_lower, dtype=np.float64))
    sqrt_p_upper = np.atleast_1d(np.asarray(sqrt_p_upper, dtype=np.float64))


    above = sqrt_p_current >= sqrt_p_upper
    below = sqrt_p_current <= sqrt_p_lower

    # Case 1: price above range → 100% token1
    v_above = L * (sqrt_p_upper - sqrt_p_lower)
    # Case 2: price below range → 100% token0, valued at current price
    v_below = external_p_current * L * (sqrt_p_upper - sqrt_p_lower) / (sqrt_p_lower * sqrt_p_upper)
    # Case 3: price in range → mix: V = x * P_ext + y
    v_in = L * (external_p_current / sqrt_p_current + sqrt_p_current - sqrt_p_lower - external_p_current / sqrt_p_upper)

    return np.where(above, v_above, np.where(below, v_below, v_in))




# Functions for collecting fees
# delta change of amount of Token A
# def delta_x(p1, p2):
#     return 1 / math.sqrt(p2) - 1 / math.sqrt(p1)


# # delta change of amount of Token B
# def delta_y(p1, p2):
#     return math.sqrt(p2) - math.sqrt(p1)
