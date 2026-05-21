"""
Side-by-side visualization for two Benchmark-format placements.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from visualize_benchmark import render_benchmark


def _load_any(path: str) -> Any:
    return torch.load(path, weights_only=False)


def compare_placements(
    left_path: str,
    right_path: str,
    output: str,
    max_nets: Optional[int] = 300,
    net_sample: str = "largest",
    show_grid: bool = False,
    show_pins: bool = False,
    show_ports: bool = True,
    show_labels: bool = False,
    net_style: str = "star",
    seed: int = 1,
    dpi: int = 150,
) -> None:
    left = _load_any(left_path)
    right = _load_any(right_path)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=dpi)

    render_benchmark(
        axes[0],
        left,
        max_nets=max_nets,
        net_sample=net_sample,
        show_grid=show_grid,
        show_pins=show_pins,
        show_ports=show_ports,
        show_labels=show_labels,
        net_style=net_style,
        seed=seed,
        title="Original",
    )

    render_benchmark(
        axes[1],
        right,
        max_nets=max_nets,
        net_sample=net_sample,
        show_grid=show_grid,
        show_pins=show_pins,
        show_ports=show_ports,
        show_labels=show_labels,
        net_style=net_style,
        seed=seed,
        title="DREAMPlace",
    )

    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    print(f"Saved comparison to {output}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two placement .pt files")
    parser.add_argument("--left", required=True, help="Original Benchmark .pt")
    parser.add_argument("--right", required=True, help="Placed Benchmark .pt")
    parser.add_argument("--output", required=True, help="Output image path")
    parser.add_argument("--max-nets", type=int, default=300, help="Max nets to draw")
    parser.add_argument("--net-sample", choices=["largest", "random"], default="largest")
    parser.add_argument("--net-style", choices=["star", "clique"], default="star")
    parser.add_argument("--show-grid", action="store_true", help="Draw placement grid")
    parser.add_argument("--show-pins", action="store_true", help="Draw macro pins")
    parser.add_argument("--hide-ports", action="store_true", help="Hide IO ports")
    parser.add_argument("--show-labels", action="store_true", help="Draw macro names")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for net sampling")
    parser.add_argument("--dpi", type=int, default=150, help="Output DPI")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    compare_placements(
        left_path=args.left,
        right_path=args.right,
        output=args.output,
        max_nets=args.max_nets,
        net_sample=args.net_sample,
        show_grid=args.show_grid,
        show_pins=args.show_pins,
        show_ports=not args.hide_ports,
        show_labels=args.show_labels,
        net_style=args.net_style,
        seed=args.seed,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
