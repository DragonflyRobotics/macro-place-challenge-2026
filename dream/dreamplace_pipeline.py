"""
End-to-end DREAMPlace pipeline for Benchmark-format data.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
	sys.path.insert(0, str(SCRIPT_DIR))

from compare_placements import compare_placements
from dreamplace_bridge import (
	export_bookshelf,
	import_bookshelf_solution,
	load_benchmark_any,
	write_dreamplace_config,
)


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
	raise FileNotFoundError(f"Could not find DREAMPlace Placer.py. Checked:\n  {checked}")


def _dreamplace_env(root: Path) -> dict[str, str]:
	env = os.environ.copy()
	pythonpath = str(root)
	if env.get("PYTHONPATH"):
		pythonpath = pythonpath + os.pathsep + env["PYTHONPATH"]
	env["PYTHONPATH"] = pythonpath
	return env


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run DREAMPlace on Benchmark-format data")
	parser.add_argument("--input", required=True, help="Benchmark .pt file")
	parser.add_argument("--work-dir", default=None, help="Working directory")
	parser.add_argument("--design-name", default=None, help="Design name override")
	parser.add_argument(
		"--dreamplace-dir",
		default=str("/DREAMPlace/install"),
		help="DREAMPlace source or install folder",
	)

	parser.add_argument("--scale", type=float, default=1000.0, help="Micron->integer scale")
	parser.add_argument("--row-height-um", type=float, default=1.0, help="Row height in microns")
	parser.add_argument("--site-width-um", type=float, default=1.0, help="Site width in microns")
	parser.add_argument("--port-size-um", type=float, default=1.0, help="Port size in microns")

	parser.add_argument("--bins-x", type=int, default=None, help="Override num_bins_x")
	parser.add_argument("--bins-y", type=int, default=None, help="Override num_bins_y")
	parser.add_argument("--iterations", type=int, default=200, help="Global placement iterations")
	parser.add_argument("--learning-rate", type=float, default=0.01, help="Global placement learning rate")
	parser.add_argument("--target-density", type=float, default=0.8, help="Target density")
	parser.add_argument("--gpu", type=int, default=0, help="Use GPU (1) or CPU (0)")
	parser.add_argument("--seed", type=int, default=123, help="Random seed")
	parser.add_argument("--macro-place-flag", type=int, default=0, help="Enable macro placement mode")

	parser.add_argument("--run", action="store_true", help="Run DREAMPlace after export")
	parser.add_argument("--compare", action="store_true", help="Render side-by-side comparison")
	parser.add_argument("--max-nets", type=int, default=300, help="Max nets to draw in comparison")
	parser.add_argument("--show-pins", action="store_true", help="Show pins in comparison")

	return parser.parse_args()


def main() -> None:
	args = _parse_args()

	input_path = Path(args.input).expanduser().resolve()
	design_name = args.design_name or input_path.stem
	work_dir = Path(args.work_dir).expanduser().resolve() if args.work_dir else SCRIPT_DIR / "dreamplace_runs" / design_name
	work_dir.mkdir(parents=True, exist_ok=True)

	dreamplace_root = _resolve_dreamplace_root(args.dreamplace_dir)

	bookshelf_dir = work_dir / "bookshelf"

	export_result = export_bookshelf(
		benchmark_path=str(input_path),
		out_dir=str(bookshelf_dir),
		design_name=design_name,
		scale=args.scale,
		row_height_um=args.row_height_um,
		site_width_um=args.site_width_um,
		include_ports=True,
		include_pin_offsets=True,
		port_size_um=args.port_size_um,
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
		num_bins_x=args.bins_x,
		num_bins_y=args.bins_y,
		iterations=args.iterations,
		learning_rate=args.learning_rate,
		target_density=args.target_density,
		gpu=args.gpu,
		seed=args.seed,
		macro_place_flag=args.macro_place_flag,
		result_dir=str((work_dir / "results").resolve()),
	)

	if args.run:
		placer = dreamplace_root / "dreamplace" / "Placer.py"
		cmd = [sys.executable, str(placer), str(config_path)]
		subprocess.run(cmd, check=True, cwd=str(dreamplace_root), env=_dreamplace_env(dreamplace_root))

	result_dir = Path(config["result_dir"]) / design_name
	pl_path = result_dir / f"{design_name}.gp.pl"
	if not pl_path.exists():
		print(f"DREAMPlace output not found at {pl_path}")
		return

	placed_pt = work_dir / f"{design_name}_dreamplace.pt"
	import_bookshelf_solution(
		original_benchmark_path=str(input_path),
		pl_path=str(pl_path),
		mapping_path=export_result["mapping"],
		output_path=str(placed_pt),
		update_fixed_from_pl=False,
	)

	if args.compare:
		output_png = work_dir / f"{design_name}_compare.png"
		compare_placements(
			left_path=str(input_path),
			right_path=str(placed_pt),
			output=str(output_png),
			max_nets=args.max_nets,
			show_pins=args.show_pins,
		)


if __name__ == "__main__":
	main()
