import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from finops import pricing, sustainability


def test_cache_break_even_reads():
    # Write cost = 0.20, read cost = 0.20, discount = 0.10 (90% off)
    # Savings per read = 0.18. Break even = 0.20 / 0.18 = 1.111...
    be = pricing.cache_break_even_reads(write_cost_per_m=0.20, read_cost_per_m=0.20, read_discount=0.10)
    assert abs(be - (0.20 / 0.18)) < 1e-6

    # 1 read is not enough (< 1.11), 2 reads is enough (> 1.11)
    assert not pricing.cache_is_worth_it(avg_cache_reads=1.0, write_cost_per_m=0.20, read_discount=0.10)
    assert pricing.cache_is_worth_it(avg_cache_reads=2.0, write_cost_per_m=0.20, read_discount=0.10)
    assert pricing.cache_is_worth_it(avg_cache_reads=3.5, write_cost_per_m=0.20, read_discount=0.10)


def test_enhanced_recommend_tier_interruption():
    # Normal spot scenario
    assert pricing.recommend_tier(hours_per_day=8, interruptible=True) == "spot"
    # Normal reserved scenario (24/7 non-interruptible)
    assert pricing.recommend_tier(hours_per_day=24, interruptible=False) == "reserved"
    # Low duty non-interruptible
    assert pricing.recommend_tier(hours_per_day=4, interruptible=False) == "on_demand"
    # Extremely high spot interruption rate (>35%) with high duty cycle -> fallback to reserved
    assert pricing.recommend_tier(hours_per_day=20, interruptible=True, interrupt_rate=0.40) == "reserved"


def test_sustainability_carbon_and_energy():
    # 1000 tokens standard query: 0.3 Wh
    wh_norm = sustainability.wh_per_query(1000, wh_per_1k_tokens=0.30, is_reasoning=False)
    assert abs(wh_norm - 0.30) < 1e-6

    # Reasoning query: 80x multiplier
    wh_reason = sustainability.wh_per_query(1000, wh_per_1k_tokens=0.30, is_reasoning=True)
    assert abs(wh_reason - 0.30 * 80.0) < 1e-6

    # Carbon in cleanest region (europe-north1 @ 30 g/kWh) vs dirtiest (europe-central2 @ 660 g/kWh)
    co2_clean = sustainability.carbon_g(1000.0, "europe-north1")  # 1 kWh = 30g
    co2_dirty = sustainability.carbon_g(1000.0, "europe-central2") # 1 kWh = 660g
    assert abs(co2_clean - 30.0) < 1e-6
    assert abs(co2_dirty - 660.0) < 1e-6
