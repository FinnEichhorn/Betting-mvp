from __future__ import annotations




def is_two_way_arb(dec1: float, dec2: float) -> tuple[bool, float]:
    inv_sum = 1 / dec1 + 1 / dec2
    return (inv_sum < 1.0), (1.0 - inv_sum)




def arb_stakes(dec1: float, dec2: float, total_stake: float) -> tuple[float, float, float, float]:
    inv1, inv2 = 1 / dec1, 1 / dec2
    s1 = total_stake * inv1 / (inv1 + inv2)
    s2 = total_stake * inv2 / (inv1 + inv2)
    ret1 = s1 * dec1
    ret2 = s2 * dec2
    guaranteed_return = (ret1 + ret2) / 2 # nearly equal
    profit = guaranteed_return - (s1 + s2)
    return s1, s2, guaranteed_return, profit