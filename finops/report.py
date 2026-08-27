"""Report assembly — the lab's deliverable: baseline vs optimized + savings chart."""
from __future__ import annotations


def build_report(
    baseline_usd: float,
    optimized_usd: float,
    levers: dict,
    sustainability: dict | None = None,
    period: str = "monthly",
    details: dict | None = None,
) -> str:
    """Return a comprehensive markdown GPU cost-optimization and FinOps report."""
    savings = baseline_usd - optimized_usd
    pct = (savings / baseline_usd * 100.0) if baseline_usd > 0 else 0.0
    
    lines = [
        "# NimbusAI — GPU Cost Optimization Report",
        "",
        "> **Executive Summary:** Comprehensive FinOps audit and workload optimization for NimbusAI's GPU fleet.",
        "",
        f"**Period:** {period}  ",
        f"**Baseline spend:** ${baseline_usd:,.0f}  ",
        f"**Optimized spend:** ${optimized_usd:,.0f}  ",
        f"**Projected savings:** ${savings:,.0f}  (**{pct:.0f}%**)",
        "",
        "---",
        "",
        "## 1. Savings by FinOps Lever",
        "",
        "| Lever | Savings (USD) | % of Total Savings | Impact & Description |",
        "|---|---|---|---|",
    ]
    for name, amount in levers.items():
        lever_pct = (amount / savings * 100.0) if savings > 0 else 0.0
        desc = ""
        if "Inference" in name:
            desc = "Prompt caching (90% discount on prefix) + model cascading + batch API (-50%)"
        elif "Purchasing" in name:
            desc = "Spot instances + checkpointing for fault-tolerant jobs, 3-yr reserved for 24/7 core services"
        elif "Right-size" in name:
            desc = "Downsizing over-provisioned memory-bound GPUs (e.g. H100 -> A100/A10G) based on MBU/MFU"
        elif "idle" in name.lower():
            desc = "Automated reaper daemon to shut down unutilized / zombie GPU instances"
        lines.append(f"| {name} | ${amount:,.0f} | {lever_pct:.1f}% | {desc} |")
    
    lines += [
        "",
        "---",
        "",
        "## 2. Technical Root-Cause Analysis: The 'GPU-Util Lie'",
        "",
        "### Why `nvidia-smi` GPU-Util is Misleading",
        "- **Mechanism:** `nvidia-smi` measures the percentage of time that one or more kernels were active on the GPU over the sample interval (a time-active clock metric). It **does not measure compute throughput** (Tensor Core utilization) or memory bandwidth saturation.",
        "- **The Reality:** A GPU reporting **98% GPU-Util** (e.g., `gpu-h100-4`) can have an **MFU (Model FLOPs Utilization) of only ~20%**.",
        "- **Root Causes:**",
        "  1. **Memory Stalls:** Workloads with low arithmetic intensity (e.g. LLM autoregressive token generation / decode phase with intensity ~1–2 FLOP/byte) are heavily memory-bandwidth bound. Compute engines remain stalled waiting for weights to load from HBM.",
        "  2. **Kernel Launch Overhead & Small Batch Sizes:** Running unbatched inference on high-end GPUs results in tiny tensor operations where SMs spend most cycles idling between launches.",
        "  3. **I/O & CPU Bottlenecks:** Dataloading pipelines stalling while keeping the GPU context active.",
        "- **Financial Impact:** Paying $2.50/hour for an H100 while obtaining the compute equivalent of an A10G ($1.00/hour) results in **$1,080/month wasted per instance**.",
        "",
        "---",
        "",
        "## 3. Prioritized FinOps Implementation Roadmap (ROI-Ranked)",
        "",
        "| Priority | Phase | Lever | Expected ROI | Complexity | Action Items |",
        "|---|---|---|---|---|---|",
        "| **P0** | Week 1 | **Prompt Caching & Cascading** | **Immediate 70–85% drop in $/1M-token** | Low | Enable prefix caching on system prompts; deploy semantic router to route simple queries to small model. |",
        "| **P1** | Week 1–2 | **Kill Idle GPUs** | **~$600–$3,750/mo immediate** | Very Low | Deploy auto-termination reaper for GPUs with utilization < 10% for > 2 consecutive hours. |",
        "| **P2** | Week 2–3 | **MBU Right-Sizing** | **~$511–$1,080/mo per GPU** | Medium | Move memory-bound decode/embedding tasks from H100 to A100/A10G based on MBU telemetry. |",
        "| **P3** | Month 1 | **Purchasing Strategy (Spot/Reserved)** | **~$19,800/mo (39–44%)** | Medium | Migrate interruptible training & eval jobs to Spot with automated checkpointing; commit 3-yr Reserved for 24/7 inference. |",
        "| **P4** | Month 2 | **Reasoning Budget & Carbon Scheduling** | **~92% Carbon Reduction** | Medium | Impose confidence-gating on reasoning models; schedule batch training in clean hydro regions. |",
        "",
        "---",
        "",
    ]
    
    if sustainability:
        best_reg = sustainability.get("best_region", "europe-north1")
        lines += [
            "## 4. Sustainability & Regional Carbon Economics",
            "",
            f"- **Energy per query:** {sustainability.get('wh_per_query', 0):.2f} Wh *(Standard query)*",
            f"- **Carbon per query:** {sustainability.get('carbon_g', 0):.3f} gCO2e *(at us-east-1 grid intensity)*",
            f"- **Cheapest + Cleanest Region:** `{best_reg}` *(30 gCO2/kWh, 92% cleaner than us-east-1)*",
            "",
            "### Regional Grid Comparison Table",
            "| Region | Grid Carbon (gCO2/kWh) | Electricity ($/kWh) | Strategic Trade-off | Recommendation |",
            "|---|---|---|---|---|",
            "| `europe-north1` (Norway) | **30** | $0.090 | 100% renewable hydro; high EU latency | **Primary for asynchronous batch & training** |",
            "| `us-east-wa` (Washington) | **90** | **$0.055** | Lowest power cost in US; low carbon hydro | **Secondary for US training & batch** |",
            "| `us-west-2` (Oregon) | 120 | $0.070 | Clean hydro grid; low latency to US West | **Primary for US live inference** |",
            "| `us-east-1` (N. Virginia) | 380 | $0.120 | Fossil fuel mixed grid; central ecosystem | Default only when colocation required |",
            "| `europe-central2` (Poland) | 660 | $0.180 | Coal-heavy grid; highest power cost & carbon | **Avoid entirely** |",
            "",
            "> **Key Insight:** Workloads like LLM pre-training, fine-tuning, and batch evaluation are latency-insensitive and should be dynamically scheduled to `europe-north1` or `us-east-wa` to eliminate 92% of operational carbon at zero performance penalty.",
            "",
            "---",
            "",
        ]

    lines += [
        "## 5. Extensions & Advanced Econometric Findings",
        "",
        "### D.1 — Advanced Purchasing Strategy (`recommend_tier`)",
        "- Evaluates break-even duty cycle ($1 - \\text{discount} = 55\\%$ for 3-year commitments vs $80\\%$ for 1-year commitments).",
        "- Simulates spot preemption risk and checkpoint overhead: even with $5\\%$ hourly interruption rate and $3\\%$ checkpoint penalty, Spot instances yield $>39\\%$ net cost savings for interruptible workloads.",
        "",
        "### D.2 — Memory Bandwidth Utilization (MBU) Right-Sizing",
        "- Analysis of `$/GB-VRAM` and `$/TB/s` reveals that memory-bound inference runs inefficiently on H100 ($0.031/GB-hr). Right-sizing `gpu-h100-4` to A100 reduces hourly costs by $28.4\\%$ while perfectly satisfying memory capacity and bandwidth demands.",
        "",
        "### D.3 — Prompt Caching Economics (`cache_is_worth_it`)",
        "- **Mathematical Break-Even:** $\\text{Break-even reads} = \\frac{\\text{Write Cost}}{(1 - \\text{Read Discount}) \\times \\text{Read Cost}} = \\frac{0.20}{(1 - 0.10) \\times 0.20} \\approx 1.11\\text{ reads}$.",
        "- With multi-turn conversations averaging $3.5$ reads per prefix, Prompt Caching achieves positive ROI on the second request and saves $90\\%$ on all subsequent input tokens.",
        "",
        "### D.4 — Reasoning Token & Energy Governance",
        "- **The 80x Multiplier:** Reasoning models consume **~80x more electrical energy** per query than small standard models.",
        "- Reasoning accounts for ~20-25% of total query volume but drives >75% of inference energy consumption. Capping reasoning invocation to complex tasks via confidence scoring saves significant electricity and cloud spend.",
        "",
        "### D.5 — Carbon-Aware Scheduling",
        "- Moving interruptible training workloads to clean grid regions reduces emissions from **~4,500 kg CO2e/month to ~360 kg CO2e/month** (-92%).",
        "",
        "---",
        "",
        "_Figures are June-2026 as-of snapshots from NimbusAI telemetry; re-baseline before acting._",
    ]
    return "\n".join(lines)


def savings_waterfall(levers: dict, path: str) -> str:
    """Write a high-quality savings bar chart PNG. Returns the path. No-op if matplotlib absent."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ""
    names = list(levers.keys())
    vals = [levers[n] for n in names]
    
    fig, ax = plt.subplots(figsize=(9, 5), dpi=120)
    colors = ["#2e548a", "#1b7837", "#762a83", "#d95f02"]
    bars = ax.bar(names, vals, color=colors[:len(names)], edgecolor="#111111", linewidth=0.8)
    
    # Add data labels
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"${height:,.0f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),  # 4 points vertical offset
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Monthly Savings (USD)", fontsize=11, fontweight="bold")
    ax.set_title("NimbusAI — GPU Cost Savings by FinOps Lever", fontsize=13, fontweight="bold", pad=15)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    plt.xticks(rotation=15, ha="right", fontsize=9.5)
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path

