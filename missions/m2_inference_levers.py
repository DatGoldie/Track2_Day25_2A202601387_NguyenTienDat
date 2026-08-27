"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Run: python missions/m2_inference_levers.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num
from finops import pricing

from finops import sustainability

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    base_cost = opt_cost = 0.0
    total_tokens = 0
    total_cached_tokens = 0
    total_input_tokens = 0
    
    # Reasoning trackers
    reasoning_reqs = non_reasoning_reqs = 0
    reasoning_tokens = non_reasoning_tokens = 0
    reasoning_opt_cost = non_reasoning_opt_cost = 0.0
    reasoning_wh = non_reasoning_wh = 0.0

    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        is_reasoning = bool(int(num(r["is_reasoning"])))
        
        toks = inp + out
        total_tokens += toks
        total_input_tokens += inp
        total_cached_tokens += cached

        # BASELINE: naive deployment — everything on the large model, no cache, no batch
        lin, lout = MODEL_PRICES["large"]
        base_cost += pricing.request_cost(inp, out, lin, lout)
        
        # OPTIMIZED: cascade (route_tier), prompt caching, batch API
        pin, pout = MODEL_PRICES[r["route_tier"]]
        req_c = pricing.request_cost(inp, out, pin, pout, cached_in=cached, batch=is_batch)
        opt_cost += req_c

        # Track energy & cost for reasoning vs normal
        eq_wh = sustainability.wh_per_query(toks, is_reasoning=is_reasoning)
        if is_reasoning:
            reasoning_reqs += 1
            reasoning_tokens += toks
            reasoning_opt_cost += req_c
            reasoning_wh += eq_wh
        else:
            non_reasoning_reqs += 1
            non_reasoning_tokens += toks
            non_reasoning_opt_cost += req_c
            non_reasoning_wh += eq_wh

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0

    # Extension 3: Cache viability metrics
    cache_hit_rate = (total_cached_tokens / total_input_tokens) if total_input_tokens else 0.0
    be_reads_small = pricing.cache_break_even_reads(0.20, 0.20, 0.10)
    be_reads_large = pricing.cache_break_even_reads(3.00, 3.00, 0.10)
    # Assume typical multi-turn cache read multiplier ~ 3.5x
    cache_profitable = pricing.cache_is_worth_it(avg_cache_reads=3.5, write_cost_per_m=0.20, read_discount=0.10)

    # Extension 4: Reasoning budget analysis
    total_reqs = len(rows)
    total_wh = reasoning_wh + non_reasoning_wh
    reasoning_traffic_pct = (reasoning_reqs / total_reqs) * 100 if total_reqs else 0
    reasoning_cost_pct = (reasoning_opt_cost / opt_cost) * 100 if opt_cost else 0
    reasoning_energy_pct = (reasoning_wh / total_wh) * 100 if total_wh else 0
    
    # Simulation: Capping reasoning to 10% traffic (from current %) via confidence gating
    target_traffic_pct = 10.0
    if reasoning_traffic_pct > target_traffic_pct:
        excess_ratio = (reasoning_traffic_pct - target_traffic_pct) / reasoning_traffic_pct
        capped_saved_cost = reasoning_opt_cost * excess_ratio * (1 - MODEL_PRICES["small"][0] / MODEL_PRICES["large"][0])
        capped_saved_wh = reasoning_wh * excess_ratio * (1 - 1.0 / sustainability.REASONING_ENERGY_MULTIPLIER)
    else:
        capped_saved_cost = 0.0
        capped_saved_wh = 0.0

    if verbose:
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")
        
        print("\n--- Extension 3: Cache Economics (cache_is_worth_it) ---")
        print(f"Total Input Tokens : {total_input_tokens:,}")
        print(f"Cached Input Tokens: {total_cached_tokens:,} ({cache_hit_rate:.1%} cache coverage)")
        print(f"Break-even reads (small): {be_reads_small:.2f} reads (saves 90% on reads)")
        print(f"Break-even reads (large): {be_reads_large:.2f} reads (saves 90% on reads)")
        print(f"Is caching profitable at 3.5 avg reads? {cache_profitable}")

        print("\n--- Extension 4: Reasoning Traffic & Energy Budget ---")
        print(f"Traffic  : Reasoning {reasoning_reqs} reqs ({reasoning_traffic_pct:.1f}%) | Normal {non_reasoning_reqs} reqs ({100-reasoning_traffic_pct:.1f}%)")
        print(f"Cost     : Reasoning ${reasoning_opt_cost:,.2f}/day ({reasoning_cost_pct:.1f}%) | Normal ${non_reasoning_opt_cost:,.2f}/day ({100-reasoning_cost_pct:.1f}%)")
        print(f"Energy   : Reasoning {reasoning_wh:,.1f} Wh ({reasoning_energy_pct:.1f}%) | Normal {non_reasoning_wh:,.1f} Wh ({100-reasoning_energy_pct:.1f}%)")
        print(f"Key Takeaway: Reasoning consumes {sustainability.REASONING_ENERGY_MULTIPLIER:.0f}x energy per token — representing {reasoning_energy_pct:.1f}% of total energy!")
        print(f"Simulation (Cap reasoning to 10% traffic): Save ${capped_saved_cost*30:,.0f}/mo and {capped_saved_wh*30/1000:,.1f} kWh/mo")

    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
        "cache_economics": {
            "cache_hit_rate": round(cache_hit_rate, 3),
            "break_even_reads": round(be_reads_small, 2),
            "is_worth_it": cache_profitable,
        },
        "reasoning_budget": {
            "traffic_pct": round(reasoning_traffic_pct, 1),
            "cost_pct": round(reasoning_cost_pct, 1),
            "energy_pct": round(reasoning_energy_pct, 1),
            "monthly_savings_if_capped_10pct": round(capped_saved_cost * 30, 2),
            "monthly_kwh_saved_if_capped_10pct": round(capped_saved_wh * 30 / 1000, 2),
        },
    }


if __name__ == "__main__":
    run()
