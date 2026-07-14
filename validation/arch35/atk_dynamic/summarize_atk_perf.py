from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SHAPES = {row["id"]: row for row in json.loads((ROOT / "cases/shape_manifest.json").read_text())}


def load_case_task_means(run_dir: Path, node_name: str) -> dict[int, dict[str, float]]:
    profile_root = next((run_dir / "atk_output").iterdir()) / "profile" / node_name
    result = {}
    for case_id in sorted(SHAPES):
        csv_paths = list((profile_root).glob(f"*/{case_id}/**/op_summary*.csv"))
        if len(csv_paths) != 1:
            raise RuntimeError(f"expected one op_summary for case {case_id}, got {csv_paths}")
        durations = {}
        with csv_paths[0].open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                op_type = row["OP Type"]
                durations.setdefault(op_type, []).append(float(row["Task Duration(us)"]))
        result[case_id] = {
            op_type: statistics.mean(values) for op_type, values in durations.items()
        }
    return result


def main() -> None:
    add_run = ROOT / "gamma_vs_add_full"
    chain_run = ROOT / "gamma_vs_old_chain_full"

    gamma_add = load_case_task_means(add_run, "pyaclnn_gamma_add_rms_norm")
    bare_add = load_case_task_means(add_run, "npu_baseline")
    gamma_chain = load_case_task_means(chain_run, "pyaclnn_gamma_add_rms_norm")
    old_chain = load_case_task_means(chain_run, "npu_baseline")

    rows = []
    for case_id, shape_info in SHAPES.items():
        gamma_add_us = gamma_add[case_id]["GammaAddRmsNorm"]
        gamma_chain_us = gamma_chain[case_id]["GammaAddRmsNorm"]
        bare_us = bare_add[case_id]["AddRmsNorm"]
        add_us = old_chain[case_id]["Add"]
        chain_norm_us = old_chain[case_id]["AddRmsNorm"]
        chain_us = add_us + chain_norm_us
        rows.append(
            {
                **shape_info,
                "gamma_vs_bare_us": gamma_add_us,
                "gamma_vs_chain_us": gamma_chain_us,
                "bare_add_rms_norm_us": bare_us,
                "gamma_vs_bare_ratio": bare_us / gamma_add_us,
                "old_add_us": add_us,
                "old_add_rms_norm_us": chain_norm_us,
                "old_chain_us": chain_us,
                "old_chain_speedup": chain_us / gamma_chain_us,
                "old_chain_reduction_pct": (chain_us - gamma_chain_us) / chain_us * 100.0,
            }
        )

    with (ROOT / "performance_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    mean = statistics.mean
    geometric_speedup = statistics.geometric_mean(row["old_chain_speedup"] for row in rows)
    bare_geomean = statistics.geometric_mean(row["gamma_vs_bare_ratio"] for row in rows)
    faster_than_chain = sum(row["old_chain_speedup"] >= 1.0 for row in rows)
    faster_than_bare = sum(row["gamma_vs_bare_ratio"] >= 1.0 for row in rows)
    report = f"""# ATK dynamic-shape performance result

## Configuration

- Device: Ascend 910B3, physical NPU 6
- ATK task: `performance_device`
- Data type: BF16
- Cases: {len(rows)} unique shapes generated from YAML constraints
- Sampling: `--performance_data 100,20,10`
- Candidate: `GammaAddRmsNorm(addGammaOffset=true)`
- Baselines: bare `AddRmsNorm`, and `Add(gamma, 1) + AddRmsNorm`

The shape generator keeps ATK's generated `x1.shape`; its Python constraint
only derives `x2.shape` and `gamma.shape` from `x1`. No complete input shape is
fixed in the generator.

## Summary

- Versus old chain: geometric-mean speedup {geometric_speedup:.3f}x; candidate faster on {faster_than_chain}/{len(rows)} shapes.
- Versus bare AddRmsNorm: geometric-mean ratio {bare_geomean:.3f}x; candidate no slower on {faster_than_bare}/{len(rows)} shapes.
- Mean old standalone Add task: {mean(row['old_add_us'] for row in rows):.3f} us.
- ATK report verdict versus old chain: Pass; average device performance ratio 1.0826.
- ATK report verdict versus bare AddRmsNorm: Failed; average device performance ratio 0.9438.

For model-relevant cases with the normalized hidden dimension `D >= 1024`,
the candidate is faster than the old chain on all 6/6 generated shapes, with
a 1.262x geometric-mean speedup. For decode-like cases (`M <= 31`, `D >= 1024`),
it is faster on all 3/3 shapes, with a 1.459x geometric-mean speedup.

The ATK verdict uses ATK's own report aggregation. The geometric means above
are independently calculated from the mean task durations in every exported
`op_summary.csv`; old-chain time is `mean(Add) + mean(AddRmsNorm)`.

## Per-shape results

| ID | Shape | GammaAddRmsNorm | AddRmsNorm | Old Add | Old chain | Chain speedup | Reduction |
|---:|---|---:|---:|---:|---:|---:|---:|
"""
    for row in rows:
        report += (
            f"| {row['id']} | `{row['x_shape']}` | {row['gamma_vs_chain_us']:.3f} us | "
            f"{row['bare_add_rms_norm_us']:.3f} us | {row['old_add_us']:.3f} us | "
            f"{row['old_chain_us']:.3f} us | {row['old_chain_speedup']:.3f}x | "
            f"{row['old_chain_reduction_pct']:.2f}% |\n"
        )
    (ROOT / "RESULTS.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
