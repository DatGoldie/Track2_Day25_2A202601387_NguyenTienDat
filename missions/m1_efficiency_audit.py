"""M1 — Efficiency Audit: MFU/MBU, the GPU-Util lie, and idle waste (deck §5).

Run: python missions/m1_efficiency_audit.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num, catalog_by_type
from finops import metrics


def run(verbose: bool = True) -> dict:
    tel = load_csv("gpu_telemetry.csv")
    cat = catalog_by_type()

    # per-row MFU/MBU, then aggregate per GPU
    agg = defaultdict(lambda: {"util": [], "mfu": [], "mbu": [], "type": None, "idle_hours": 0})
    for r in tel:
        gtype = r["gpu_type"]
        peak_fp16 = num(cat[gtype]["peak_tflops_fp16"])
        peak_bw = num(cat[gtype]["peak_bw_tbs"])
        mfu = metrics.compute_mfu(num(r["achieved_tflops"]), peak_fp16)
        mbu = metrics.compute_mbu(num(r["achieved_bw_tbs"]), peak_bw)
        a = agg[r["gpu_id"]]
        a["type"] = gtype
        a["util"].append(num(r["gpu_util_pct"]))
        a["mfu"].append(mfu)
        a["mbu"].append(mbu)
        if num(r["gpu_util_pct"]) < 10:  # effectively idle this interval (1h)
            a["idle_hours"] += 1

    summary = []
    for gid, a in agg.items():
        summary.append({
            "gpu_id": gid, "gpu_type": a["type"],
            "gpu_util_pct": round(sum(a["util"]) / len(a["util"]), 1),
            "mfu": round(sum(a["mfu"]) / len(a["mfu"]), 3),
            "mbu": round(sum(a["mbu"]) / len(a["mbu"]), 3),
            "idle_hours": a["idle_hours"],
        })

    lies = metrics.flag_util_lies(summary)
    idle_waste = 0.0
    for s in summary:
        on_demand = num(catalog_by_type()[s["gpu_type"]]["on_demand_hr"])
        idle_waste += metrics.idle_waste_usd(s["idle_hours"], on_demand)

    # --- Extension 2: Right-sizing analysis based on MBU & $/GB-VRAM ---
    rightsizing_analysis = []
    total_rightsize_monthly_savings = 0.0
    for l in lies:
        cur_type = l["gpu_type"]
        cur_price = num(cat[cur_type]["on_demand_hr"])
        cur_bw = num(cat[cur_type]["peak_bw_tbs"])
        cur_vram = num(cat[cur_type]["hbm_gb"])
        
        # Proposed replacement: if H100 with low MFU/MBU, rightsize to A100 or A10G
        rec_type = "A100" if cur_type in ("H100", "H200") else "L4"
        rec_price = num(cat[rec_type]["on_demand_hr"])
        rec_bw = num(cat[rec_type]["peak_bw_tbs"])
        rec_vram = num(cat[rec_type]["hbm_gb"])
        
        hourly_saved = cur_price - rec_price
        monthly_saved = hourly_saved * 24 * 30
        total_rightsize_monthly_savings += monthly_saved
        
        rightsizing_analysis.append({
            "gpu_id": l["gpu_id"],
            "current_type": cur_type,
            "current_cost_hr": cur_price,
            "recommended_type": rec_type,
            "recommended_cost_hr": rec_price,
            "savings_pct": round((hourly_saved / cur_price) * 100, 1),
            "monthly_savings_usd": round(monthly_saved, 2),
            "reason": f"MBU={l['mbu']:.2f}, MFU={l['mfu']:.2f} fits {rec_type} (BW: {rec_bw} TB/s, VRAM: {rec_vram}GB)",
        })

    if verbose:
        print("== M1 Efficiency Audit ==")
        print(f"{'GPU':14}{'type':7}{'util%':>7}{'MFU':>7}{'MBU':>7}{'idle_h':>8}")
        for s in sorted(summary, key=lambda x: x["mfu"]):
            print(f"{s['gpu_id']:14}{s['gpu_type']:7}{s['gpu_util_pct']:>7}{s['mfu']:>7}{s['mbu']:>7}{s['idle_hours']:>8}")
        print(f"\nGPU-Util LIES (util>=90% but MFU<30%): {[l['gpu_id'] for l in lies]}")
        print(f"Idle waste (1 day): ${idle_waste:,.2f}  ->  ${idle_waste*30:,.0f}/month")
        
        print("\n--- Extension 2: Right-Sizing Analysis (MBU & $/GB-VRAM) ---")
        print(f"{'GPU Catalog':12}{'Price/hr':>10}{'VRAM(GB)':>10}{'BW(TB/s)':>10}{'$/GB-VRAM':>12}{'$/TB/s BW':>12}")
        for gt, cdata in cat.items():
            price = num(cdata["on_demand_hr"])
            vram = num(cdata["hbm_gb"])
            bw = num(cdata["peak_bw_tbs"])
            cost_per_gb = price / vram if vram else 0
            cost_per_bw = price / bw if bw else 0
            print(f"{gt:12}${price:>9.2f}{vram:>10.0f}{bw:>10.1f}${cost_per_gb:>11.4f}${cost_per_bw:>11.2f}")

        print("\nRight-Sizing Recommendations for Util-Lies / Memory-Bound GPUs:")
        for r in rightsizing_analysis:
            print(f"  * {r['gpu_id']} ({r['current_type']} @ ${r['current_cost_hr']}/hr) -> {r['recommended_type']} (@ ${r['recommended_cost_hr']}/hr): Save {r['savings_pct']}% (${r['monthly_savings_usd']:,.0f}/mo) | {r['reason']}")
        print(f"Total Right-Sizing Monthly Savings Potential: ${total_rightsize_monthly_savings:,.0f}/month")

    return {
        "summary": summary,
        "lies": lies,
        "idle_waste_daily": round(idle_waste, 2),
        "rightsizing_analysis": rightsizing_analysis,
        "rightsize_monthly_savings": round(total_rightsize_monthly_savings, 2),
    }


if __name__ == "__main__":
    run()
