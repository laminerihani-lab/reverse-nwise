"""Emit LaTeX tables and CSV from experiment result JSON files.

Turns the genuine measured results (exp01/exp03/exp04/exp02) into paper-ready
LaTeX table bodies and CSVs, so the empirical section is regenerated directly
from the artifacts (no hand-transcription).

Usage:
    python -m analysis.tables --exp04 results/exp04_30runs.json --out results/tables/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}\\%"


_METRIC_LABEL = {
    "OCov_s": "$\\mathrm{OCov}_s$",
    "eta_s": "$\\eta_s$",
    "FDR_pct": "FDR",
    "FDR": "FDR",
}


def _num(n: int) -> str:
    """Thousands-separated integer, LaTeX-safe (braces protect the comma)."""
    return f"{n:,}".replace(",", "{,}")


def _metric_label(name: str) -> str:
    """Render a raw metric key as a LaTeX-safe label (math where needed)."""
    return _METRIC_LABEL.get(name, name.replace("_", "\\_"))


def table_main_comparison(exp04: dict) -> str:
    """Main results table: per-method OCov_s / FDR with 95% CIs (from exp04)."""
    agg = exp04["aggregate"]
    lines = [
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Method & Tests & OCov$_s$ (95\\% CI) & FDR (95\\% CI) \\\\",
        "\\midrule",
    ]
    # keep RNWise first, then baselines
    order = ["Reverse N-Wise"] + [m for m in agg if m != "Reverse N-Wise"]
    for m in order:
        a = agg[m]
        o, f = a["OCov_s"], a["FDR_pct"]
        ocov = f"{_fmt_pct(o['mean'])} [{o['ci95_lo']*100:.1f}, {o['ci95_hi']*100:.1f}]"
        fdr = f"{_fmt_pct(f['mean'])} [{f['ci95_lo']*100:.1f}, {f['ci95_hi']*100:.1f}]"
        name = m if m != "Reverse N-Wise" else "\\textbf{Reverse N-Wise}"
        lines.append(f"{name} & {a['tests_mean']:.0f} & {ocov} & {fdr} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def table_significance(exp04: dict) -> str:
    """Pairwise MWU + Cliff's delta table (from exp04)."""
    lines = [
        "\\begin{tabular}{llrrl}",
        "\\toprule",
        "Metric & vs.\\ Baseline & $p$-value & $\\delta$ & Effect \\\\",
        "\\midrule",
    ]
    for c in exp04["comparisons"]:
        p = c["p_value"]
        pstr = f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"
        sig = "$^{*}$" if c["significant"] else ""
        lines.append(
            f"{_metric_label(c['metric'])} & {c['b']} & {pstr}{sig} & {c['cliffs_delta']:.2f} & {c['effect']} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def table_strength(exp03: dict) -> str:
    """Strength-scaling table (from exp03)."""
    lines = [
        "\\begin{tabular}{rrrrrrl}",
        "\\toprule",
        "$s$ & Universe & Feasible & Tests & OCov$_s$ & $v^s$ & Bound \\\\",
        "\\midrule",
    ]
    for r in exp03["by_strength"]:
        ok = "\\checkmark" if r["bound_satisfied"] else "\\times"
        lines.append(
            f"{r['s']} & {r['full_universe']} & {r['feasible_tuples']} & "
            f"{r['prioritized_tests']} & {_fmt_pct(r['OCov_s'])} & "
            f"{r['corollary1_lower_bound']} & ${ok}$ \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def table_multidomain(exp02: dict) -> str:
    """Multi-domain generality table (from exp02): RNWise vs random per domain."""
    lines = [
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Domain & Opt. & Method & Tests & OCov$_s$ & FDR \\\\",
        "\\midrule",
    ]
    pretty = {"quantum": "Quantum", "vision": "Vision", "nlp": "NLP"}
    for dom, r in exp02["domains"].items():
        dname = pretty.get(dom, dom)
        for i, row in enumerate(r["results"]):
            method = "\\textbf{Reverse N-Wise}" if row["method"].startswith("Reverse") else "Random"
            dcell = dname if i == 0 else ""
            ocell = r["optimizer"] if i == 0 else ""
            lines.append(
                f"{dcell} & {ocell} & {method} & {row['tests']} & "
                f"{_fmt_pct(row['OCov_s'])} & {row['FDR']} \\\\"
            )
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"  # replace trailing midrule
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def table_cost(exp06: dict) -> str:
    """Cost-profile table (from exp06): per-stage timing + budget."""
    t = exp06["timings_s"]
    br = exp06["budget_ratio"]
    lines = [
        "\\begin{tabular}{lr}",
        "\\toprule",
        "Quantity & Value \\\\",
        "\\midrule",
        f"Stage 1 (feasibility) & {t['step1_feasibility']:.2f} s \\\\",
        f"Stage 2 (covering array) & {t['step2_covering_array']:.2f} s \\\\",
        f"Stage 3 (inverse mapping) & {t['step3_inverse_map']:.2f} s \\\\",
        f"Stage 4 (prioritise) & {t['step4_prioritize']:.2f} s \\\\",
        f"Pipeline total & {t['pipeline_total']:.2f} s \\\\",
        f"Peak memory & {exp06['peak_memory_mb']:.1f} MB \\\\",
        f"SUT evaluations & {_num(br['rnwise_calls'])} \\\\",
        f"Fraction of exhaustive sweep & {br['fraction_of_exhaustive']*100:.2f}\\% \\\\",
        "\\bottomrule",
        "\\end{tabular}",
    ]
    return "\n".join(lines)


def table_ablation(exp05: dict) -> str:
    """Three-panel ablation table (from exp05): optimizer, granularity, lambda.

    Optimizer and lambda panels run without warm start on a fixed 40-target
    sample (raw inverse-map power); the granularity panel uses warm start over
    the full feasible set.
    """
    lines = [
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "\\multicolumn{4}{l}{\\emph{(a) Inverse-map optimizer (no warm start, 40 targets)}} \\\\",
        "Optimizer & Realise & $\\mathrm{OCov}_s$ & SUT evals \\\\",
        "\\midrule",
    ]
    for r in exp05["optimizer_ablation"]:
        lines.append(
            f"{r['optimizer']} & {_fmt_pct(r['realisation_rate'])} & "
            f"{_fmt_pct(r['OCov_s'])} & {_num(r['sut_evals'])} \\\\"
        )
    lines += [
        "\\midrule",
        "\\multicolumn{4}{l}{\\emph{(b) Output granularity (warm start, full set)}} \\\\",
        "Bins/factor & Universe & Feasible & $\\mathrm{OCov}_s$ \\\\",
        "\\midrule",
    ]
    for r in exp05["granularity_ablation"]:
        lines.append(
            f"{r['levels']} & {r['full_universe']} & {r['feasible_tuples']} & "
            f"{_fmt_pct(r['OCov_s'])} \\\\"
        )
    lines += [
        "\\midrule",
        "\\multicolumn{4}{l}{\\emph{(c) Regulariser weight $\\lambda$ (no warm start, 40 targets)}} \\\\",
        "$\\lambda$ & Realise & $\\mathrm{OCov}_s$ & SUT evals \\\\",
        "\\midrule",
    ]
    for r in exp05["lambda_ablation"]:
        lines.append(
            f"{r['lambda']:g} & {_fmt_pct(r['realisation_rate'])} & "
            f"{_fmt_pct(r['OCov_s'])} & {_num(r['sut_evals'])} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def csv_main_comparison(exp04: dict) -> str:
    agg = exp04["aggregate"]
    rows = ["method,tests,OCov_s_mean,OCov_s_lo,OCov_s_hi,FDR_mean,FDR_lo,FDR_hi"]
    for m, a in agg.items():
        o, f = a["OCov_s"], a["FDR_pct"]
        rows.append(
            f"{m},{a['tests_mean']},{o['mean']},{o['ci95_lo']},{o['ci95_hi']},"
            f"{f['mean']},{f['ci95_lo']},{f['ci95_hi']}"
        )
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="emit LaTeX/CSV tables from results")
    ap.add_argument("--exp04", type=str, default="")
    ap.add_argument("--exp03", type=str, default="")
    ap.add_argument("--exp02", type=str, default="")
    ap.add_argument("--exp05", type=str, default="")
    ap.add_argument("--exp06", type=str, default="")
    ap.add_argument("--out", type=str, default="results/tables")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.exp04:
        e4 = json.loads(Path(args.exp04).read_text())
        (out / "table_main.tex").write_text(table_main_comparison(e4))
        (out / "table_significance.tex").write_text(table_significance(e4))
        (out / "main_comparison.csv").write_text(csv_main_comparison(e4))
        print(f"wrote {out}/table_main.tex, table_significance.tex, main_comparison.csv")

    if args.exp03:
        e3 = json.loads(Path(args.exp03).read_text())
        (out / "table_strength.tex").write_text(table_strength(e3))
        print(f"wrote {out}/table_strength.tex")

    if args.exp02:
        e2 = json.loads(Path(args.exp02).read_text())
        (out / "table_multidomain.tex").write_text(table_multidomain(e2))
        print(f"wrote {out}/table_multidomain.tex")

    if args.exp06:
        e6 = json.loads(Path(args.exp06).read_text())
        (out / "table_cost.tex").write_text(table_cost(e6))
        print(f"wrote {out}/table_cost.tex")

    if args.exp05:
        e5 = json.loads(Path(args.exp05).read_text())
        (out / "table_ablation.tex").write_text(table_ablation(e5))
        print(f"wrote {out}/table_ablation.tex")


if __name__ == "__main__":
    main()
