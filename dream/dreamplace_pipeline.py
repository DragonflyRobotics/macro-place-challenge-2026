"""
End-to-end DREAMPlace pipeline for Benchmark-format data.
"""

from __future__ import annotations
from .dreamplace_bridge import (
    export_bookshelf,
    import_bookshelf_solution,
    load_benchmark_any,
    write_dreamplace_config,
)
from .compare_placements import compare_placements

import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


WORKSPACE_DIR = SCRIPT_DIR.parents[1]


def _resolve_dreamplace_root(dreamplace_dir: str) -> Path:
    """Resolve a DREAMPlace source or install directory."""
    candidate = Path(dreamplace_dir).expanduser()
    if candidate.is_absolute():
        candidates = [candidate]
    else:
        candidates = [
            (Path.cwd() / candidate).resolve(),
            (SCRIPT_DIR / candidate).resolve(),
            (SCRIPT_DIR.parent / candidate).resolve(),
            (WORKSPACE_DIR / candidate).resolve(),
        ]

    for root in candidates:
        if (root / "install" / "dreamplace" / "Placer.py").exists():
            return root / "install"
        if (root / "dreamplace" / "Placer.py").exists():
            return root

    checked = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not find DREAMPlace Placer.py. Checked:\n  {checked}"
    )


def _dreamplace_env(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = str(root)
    if env.get("PYTHONPATH"):
        pythonpath = pythonpath + os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath
    return env


INPUT = "run.pt"
WORK_DIR = "/DREAMPlace/runs"
DESIGN_NAME = "run"
DREAMPLACE_DIR = "/DREAMPlace/install"

SCALE = 1000.0
ROW_HEIGHT_UM = 1.0
SITE_WIDTH_UM = 1.0
PORT_SIZE_UM = 1.0

BINS_X = 256
BINS_Y = 256
ITERATIONS = 600
LEARNING_RATE = 0.01
TARGET_DENSITY = 0.4
GPU = 1
SEED = 123
MACRO_PLACE_FLAG = 0

RUN = True
COMPARE = False
MAX_NETS = 300
SHOW_PINS = False


def main() -> None:
    input_path = Path(INPUT).expanduser().resolve()
    design_name = DESIGN_NAME or input_path.stem

    work_dir = (
        Path(WORK_DIR).expanduser().resolve()
        if WORK_DIR
        else SCRIPT_DIR / "dreamplace_runs" / design_name
    )
    work_dir.mkdir(parents=True, exist_ok=True)

    dreamplace_root = _resolve_dreamplace_root(DREAMPLACE_DIR)

    bookshelf_dir = work_dir / "bookshelf"

    export_result = export_bookshelf(
        benchmark_path=str(input_path),
        out_dir=str(bookshelf_dir),
        design_name=design_name,
        scale=SCALE,
        row_height_um=ROW_HEIGHT_UM,
        site_width_um=SITE_WIDTH_UM,
        include_ports=True,
        include_pin_offsets=True,
        port_size_um=PORT_SIZE_UM,
    )

    config_path = work_dir / f"{design_name}.config.json"

    bench_data = load_benchmark_any(str(input_path))

    if isinstance(bench_data, dict):
        canvas_w = float(bench_data["canvas_width"])
        canvas_h = float(bench_data["canvas_height"])
        grid_rows = int(bench_data.get("grid_rows", 0))
        grid_cols = int(bench_data.get("grid_cols", 0))
    else:
        canvas_w = float(bench_data.canvas_width)
        canvas_h = float(bench_data.canvas_height)
        grid_rows = int(bench_data.grid_rows)
        grid_cols = int(bench_data.grid_cols)

    config = write_dreamplace_config(
        config_path=str(config_path),
        aux_path=export_result["aux"],
        canvas_w=canvas_w,
        canvas_h=canvas_h,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        num_bins_x=BINS_X,
        num_bins_y=BINS_Y,
        iterations=ITERATIONS,
        learning_rate=LEARNING_RATE,
        target_density=TARGET_DENSITY,
        gpu=GPU,
        seed=SEED,
        macro_place_flag=MACRO_PLACE_FLAG,
        result_dir=str((work_dir / "results").resolve()),
    )

    if RUN:
        placer = dreamplace_root / "dreamplace" / "Placer.py"

        cmd = [sys.executable, str(placer), str(config_path)]

        p = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        for line in p.stdout:
            print(line, end="")  # live output

        p.wait()

    result_dir = Path(config["result_dir"]) / design_name
    pl_path = result_dir / f"{design_name}.gp.pl"

    if not pl_path.exists():
        print(f"DREAMPlace output not found at {pl_path}")
        return

    placed_pt = work_dir / f"{design_name}_dreamplace.pt"

    return import_bookshelf_solution(
        original_benchmark_path=str(input_path),
        pl_path=str(pl_path),
        mapping_path=export_result["mapping"],
        output_path=str(placed_pt),
        update_fixed_from_pl=False,
    )
