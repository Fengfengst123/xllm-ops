from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


def prepare(gamma_case_path: Path, output_dir: Path) -> None:
    gamma_cases = json.loads(gamma_case_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)

    versus_add_cases = []
    versus_chain_cases = []
    shape_rows = []
    for case_id, gamma_case in enumerate(gamma_cases):
        inputs = {item["name"]: item for item in gamma_case["inputs"]}
        shape_rows.append(
            {
                "id": case_id,
                "dtype": inputs["x1"]["dtype"],
                "x_shape": inputs["x1"]["shape"],
                "gamma_shape": inputs["gamma"]["shape"],
            }
        )

        versus_add_case = copy.deepcopy(gamma_case)
        versus_add_case["id"] = case_id
        versus_add_case["api_type"] = "bare_add_rms_norm_perf"
        versus_add_cases.append(versus_add_case)

        versus_chain_case = copy.deepcopy(gamma_case)
        versus_chain_case["id"] = case_id
        versus_chain_case["api_type"] = "add_plus_add_rms_norm_perf"
        versus_chain_cases.append(versus_chain_case)

    (output_dir / "gamma_cases.json").write_text(
        json.dumps(gamma_cases, indent=2), encoding="utf-8"
    )
    (output_dir / "gamma_vs_add_rms_norm_cases.json").write_text(
        json.dumps(versus_add_cases, indent=2), encoding="utf-8"
    )
    (output_dir / "gamma_vs_old_chain_cases.json").write_text(
        json.dumps(versus_chain_cases, indent=2), encoding="utf-8"
    )
    (output_dir / "shape_manifest.json").write_text(
        json.dumps(shape_rows, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: prepare_aligned_cases.py GAMMA_CASE_JSON OUTPUT_DIR")
    prepare(Path(sys.argv[1]), Path(sys.argv[2]))
