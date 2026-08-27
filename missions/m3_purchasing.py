"""M3 — Purchasing Strategy: break-even, tier choice, spot-checkpoint sim (deck §4).

Run: python missions/m3_purchasing.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import pricing

from finops import sustainability

DAYS = 30


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    on_demand_monthly = optimized_monthly = 0.0
    recs = []
    
    # Trackers for carbon-aware scheduling (Extension 5)
    interruptible_jobs = []
    total_interruptible_wh = 0.0

    for j in jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        interruptible = bool(int(num(j["interruptible"])))
        c = cat[gtype]
        gpu_hours = hpd * DAYS * ngpu
        od = num(c["on_demand_hr"])
        on_demand_cost = gpu_hours * od

        tier = pricing.recommend_tier(hpd, interruptible, gpu_type=gtype)
        if tier == "spot":
            sim = pricing.spot_checkpoint_cost(gpu_hours, num(c["spot_hr"]), od)
            opt_cost = sim["spot_cost"]
        elif tier == "reserved":
            opt_cost = gpu_hours * num(c["reserved_3yr_hr"])
        else:
            opt_cost = on_demand_cost

        on_demand_monthly += on_demand_cost
        optimized_monthly += opt_cost
        recs.append({"job_id": j["job_id"], "gpu_type": gtype, "tier": tier,
                     "on_demand": round(on_demand_cost), "optimized": round(opt_cost)})

        if interruptible:
            watts = num(c["watts"])
            job_wh = gpu_hours * watts
            total_interruptible_wh += job_wh
            interruptible_jobs.append({
                "job_id": j["job_id"], "gpu_type": gtype, "gpu_hours": gpu_hours,
                "watts": watts, "wh": job_wh
            })

    savings = on_demand_monthly - optimized_monthly
    savings_pct = savings / on_demand_monthly * 100 if on_demand_monthly else 0.0

    # Extension 5: Multi-region carbon and energy cost breakdown
    regional_carbon_analysis = {}
    for reg, gco2 in sustainability.REGION_CARBON.items():
        co2_kg = sustainability.carbon_g(total_interruptible_wh, reg) / 1000.0
        elec_usd = sustainability.energy_cost_usd(total_interruptible_wh, reg)
        regional_carbon_analysis[reg] = {
            "gco2_kwh": gco2,
            "price_kwh": sustainability.REGION_PRICE_KWH.get(reg, 0.12),
            "monthly_co2_kg": round(co2_kg, 1),
            "monthly_elec_usd": round(elec_usd, 2),
        }
    
    base_co2 = regional_carbon_analysis["us-east-1"]["monthly_co2_kg"]
    clean_co2 = regional_carbon_analysis["europe-north1"]["monthly_co2_kg"]
    co2_saved_kg = base_co2 - clean_co2
    co2_saved_pct = (co2_saved_kg / base_co2) * 100 if base_co2 else 0

    if verbose:
        print("== M3 Purchasing Strategy ==")
        print(f"break-even utilization @ 45% reserved discount = {pricing.break_even_utilization(0.45):.0%}")
        print(f"{'job':18}{'gpu':7}{'tier':11}{'on-demand':>12}{'optimized':>12}")
        for r in recs:
            print(f"{r['job_id']:18}{r['gpu_type']:7}{r['tier']:11}${r['on_demand']:>11,}${r['optimized']:>11,}")
        print(f"\nmonthly: on-demand ${on_demand_monthly:,.0f} -> optimized ${optimized_monthly:,.0f}  ({savings_pct:.1f}% saved)")

        print("\n--- Extension 5: Carbon-Aware Workload Scheduling ---")
        print(f"Total Interruptible Compute Energy: {total_interruptible_wh/1000:,.1f} kWh/month across {len(interruptible_jobs)} batch jobs")
        print(f"{'Region':18}{'gCO2/kWh':>10}{'$/kWh':>10}{'Monthly kgCO2e':>16}{'Elec Cost ($)':>14}")
        for reg, rdata in regional_carbon_analysis.items():
            print(f"{reg:18}{rdata['gco2_kwh']:>10}{rdata['price_kwh']:>10.3f}{rdata['monthly_co2_kg']:>16,.1f}${rdata['monthly_elec_usd']:>13,.2f}")
        print(f"Shift to Clean Region (europe-north1): Save {co2_saved_kg:,.1f} kg CO2e/mo ({co2_saved_pct:.1f}% reduction in emissions!)")

    return {
        "recommendations": recs,
        "on_demand_monthly": round(on_demand_monthly),
        "optimized_monthly": round(optimized_monthly),
        "savings_pct": round(savings_pct, 1),
        "regional_carbon": regional_carbon_analysis,
        "carbon_savings_kg": round(co2_saved_kg, 1),
    }


if __name__ == "__main__":
    run()
