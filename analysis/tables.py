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
            f"{c['metric']} & {c['b']} & {pstr}{sig} & {c['cliffs_delta']:.2f} & {c['effect']} \\\\"
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


if __name__ == "__main__":
    main()
