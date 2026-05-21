"""
Visualizer for Benchmark-format macro placement data.

This script is format-driven: it only expects fields from the Benchmark tensor
schema and renders placement, nets, pins, and ports regardless of data source.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

try:
    from .generate_benchmark import Benchmark
except Exception:
    Benchmark = None  # type: ignore[assignment]


@dataclass
class BenchmarkView:
    name: str
    canvas_width: float
    canvas_height: float
    num_macros: int
    num_hard_macros: int
    num_soft_macros: int
    macro_positions: torch.Tensor
    macro_sizes: torch.Tensor
    macro_fixed: torch.Tensor
    macro_names: List[str]
    num_nets: int
    net_nodes: List[torch.Tensor]
    net_weights: torch.Tensor
    grid_rows: int
    grid_cols: int
    port_positions: torch.Tensor
    macro_pin_offsets: List[torch.Tensor]
    net_pin_nodes: List[torch.Tensor]


def _get_field(data: Any, key: str, default: Any = None) -> Any:
    if isinstance(data, dict):
        return data.get(key, default)
    return getattr(data, key, default)


def _coerce_tensor(value: Any, dtype: torch.dtype, shape: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
    t = torch.as_tensor(value, dtype=dtype)
    if shape is not None and t.shape != shape:
        raise ValueError(f"Expected shape {shape} for tensor, got {tuple(t.shape)}")
    return t


def _coerce_list_of_tensors(items: Any, dtype: torch.dtype) -> List[torch.Tensor]:
    if items is None:
        return []
    out: List[torch.Tensor] = []
    for item in items:
        out.append(torch.as_tensor(item, dtype=dtype))
    return out


def _load_any(path: str) -> Any:
    data = torch.load(path, weights_only=False)
    return data


def _normalize_benchmark(data: Any) -> BenchmarkView:
    if Benchmark is not None and isinstance(data, Benchmark):
        raw = data
    else:
        raw = data

    num_macros = int(_get_field(raw, "num_macros"))
    num_hard = int(_get_field(raw, "num_hard_macros", num_macros))
    num_soft = int(_get_field(raw, "num_soft_macros", num_macros - num_hard))

    macro_positions = _coerce_tensor(_get_field(raw, "macro_positions"), torch.float32)
    macro_sizes = _coerce_tensor(_get_field(raw, "macro_sizes"), torch.float32)
    macro_fixed = _coerce_tensor(_get_field(raw, "macro_fixed"), torch.bool)

    macro_names = _get_field(raw, "macro_names", None)
    if macro_names is None:
        macro_names = [f"M{i:04d}" for i in range(num_macros)]

    net_nodes = _coerce_list_of_tensors(_get_field(raw, "net_nodes", []), torch.long)
    net_pin_nodes = _coerce_list_of_tensors(_get_field(raw, "net_pin_nodes", []), torch.long)
    net_weights = _coerce_tensor(_get_field(raw, "net_weights", torch.ones(len(net_nodes))), torch.float32)

    port_positions = _coerce_tensor(_get_field(raw, "port_positions", torch.zeros(0, 2)), torch.float32)
    macro_pin_offsets = _coerce_list_of_tensors(_get_field(raw, "macro_pin_offsets", []), torch.float32)

    return BenchmarkView(
        name=str(_get_field(raw, "name", "benchmark")),
        canvas_width=float(_get_field(raw, "canvas_width", float(macro_positions[:, 0].max().item()))),
        canvas_height=float(_get_field(raw, "canvas_height", float(macro_positions[:, 1].max().item()))),
        num_macros=num_macros,
        num_hard_macros=num_hard,
        num_soft_macros=num_soft,
        macro_positions=macro_positions,
        macro_sizes=macro_sizes,
        macro_fixed=macro_fixed,
        macro_names=list(macro_names),
        num_nets=int(_get_field(raw, "num_nets", len(net_nodes))),
        net_nodes=net_nodes,
        net_weights=net_weights,
        grid_rows=int(_get_field(raw, "grid_rows", 0)),
        grid_cols=int(_get_field(raw, "grid_cols", 0)),
        port_positions=port_positions,
        macro_pin_offsets=macro_pin_offsets,
        net_pin_nodes=net_pin_nodes,
    )


def _macro_rects(positions: torch.Tensor, sizes: torch.Tensor) -> torch.Tensor:
    half = sizes / 2.0
    return torch.cat([positions - half, sizes], dim=1)


def _pin_positions_for_macro(
    macro_idx: int,
    pin_idx: int,
    positions: torch.Tensor,
    macro_pin_offsets: Sequence[torch.Tensor],
) -> torch.Tensor:
    center = positions[macro_idx]
    if macro_idx < len(macro_pin_offsets) and pin_idx < len(macro_pin_offsets[macro_idx]):
        return center + macro_pin_offsets[macro_idx][pin_idx]
    return center


def _endpoint_positions(
    view: BenchmarkView,
    nodes: torch.Tensor,
    use_pins: bool,
) -> torch.Tensor:
    if use_pins and view.net_pin_nodes:
        pts = []
        for owner_idx, pin_idx in nodes.tolist():
            if owner_idx < view.num_macros:
                pt = _pin_positions_for_macro(owner_idx, pin_idx, view.macro_positions, view.macro_pin_offsets)
                pts.append(pt)
            else:
                port_idx = owner_idx - view.num_macros
                if 0 <= port_idx < len(view.port_positions):
                    pts.append(view.port_positions[port_idx])
        if pts:
            return torch.stack(pts, dim=0)
        return torch.zeros(0, 2)

    pts = []
    for idx in nodes.tolist():
        if 0 <= idx < view.num_macros:
            pts.append(view.macro_positions[idx])
    if pts:
        return torch.stack(pts, dim=0)
    return torch.zeros(0, 2)


def _choose_nets(
    net_nodes: Sequence[torch.Tensor],
    net_weights: torch.Tensor,
    max_nets: Optional[int],
    mode: str,
    seed: int,
) -> List[int]:
    total = len(net_nodes)
    if max_nets is None or max_nets >= total:
        return list(range(total))

    if mode == "largest":
        sizes = torch.tensor([len(n) for n in net_nodes], dtype=torch.float32)
        scores = sizes * net_weights[: len(net_nodes)]
        order = torch.argsort(scores, descending=True)
        return order[:max_nets].tolist()

    rng = random.Random(seed)
    idxs = list(range(total))
    rng.shuffle(idxs)
    return idxs[:max_nets]


def _render_view(
    ax: plt.Axes,
    view: BenchmarkView,
    max_nets: Optional[int],
    net_sample: str,
    show_grid: bool,
    show_pins: bool,
    show_ports: bool,
    show_labels: bool,
    net_style: str,
    seed: int,
    title: Optional[str],
) -> None:
    rects = _macro_rects(view.macro_positions, view.macro_sizes)
    canvas_w = view.canvas_width
    canvas_h = view.canvas_height

    # Canvas boundary
    ax.add_patch(
        plt.Rectangle(
            (0.0, 0.0),
            canvas_w,
            canvas_h,
            facecolor="none",
            edgecolor="black",
            linewidth=1.4,
            linestyle="--",
        )
    )

    # Optional grid
    if show_grid and view.grid_rows > 0 and view.grid_cols > 0:
        for r in range(1, view.grid_rows):
            y = canvas_h * r / view.grid_rows
            ax.plot([0.0, canvas_w], [y, y], color="#DDDDDD", linewidth=0.5, zorder=0)
        for c in range(1, view.grid_cols):
            x = canvas_w * c / view.grid_cols
            ax.plot([x, x], [0.0, canvas_h], color="#DDDDDD", linewidth=0.5, zorder=0)

    # Macro colors
    hard_color = "#4C78A8"
    soft_color = "#A6A6A6"
    fixed_border = "#E45756"

    for i, (x, y, w, h) in enumerate(rects.tolist()):
        is_hard = i < view.num_hard_macros
        is_fixed = bool(view.macro_fixed[i].item())

        face = hard_color if is_hard else soft_color
        edge = fixed_border if is_fixed else "black"
        lw = 1.6 if is_fixed else 0.8

        ax.add_patch(
            plt.Rectangle(
                (x, y),
                w,
                h,
                facecolor=face,
                edgecolor=edge,
                linewidth=lw,
                alpha=0.75,
                zorder=2,
            )
        )

        if show_labels:
            ax.text(
                x + w / 2.0,
                y + h / 2.0,
                view.macro_names[i],
                fontsize=6,
                ha="center",
                va="center",
                color="black",
            )

    # Ports
    if show_ports and len(view.port_positions) > 0:
        ax.scatter(
            view.port_positions[:, 0].numpy(),
            view.port_positions[:, 1].numpy(),
            s=18,
            c="#59A14F",
            marker="s",
            zorder=3,
            linewidths=0.3,
            edgecolors="black",
        )

    # Pins
    if show_pins and view.macro_pin_offsets:
        pin_pts = []
        for macro_idx in range(min(view.num_hard_macros, len(view.macro_pin_offsets))):
            offsets = view.macro_pin_offsets[macro_idx]
            if offsets.numel() == 0:
                continue
            pts = view.macro_positions[macro_idx].unsqueeze(0) + offsets
            pin_pts.append(pts)
        if pin_pts:
            pins = torch.cat(pin_pts, dim=0)
            ax.scatter(
                pins[:, 0].numpy(),
                pins[:, 1].numpy(),
                s=8,
                c="#F28E2B",
                marker="o",
                zorder=4,
                linewidths=0.2,
                edgecolors="black",
            )

    # Nets
    net_indices = _choose_nets(view.net_nodes, view.net_weights, max_nets, net_sample, seed)
    use_pins = bool(view.net_pin_nodes)
    for idx in net_indices:
        if idx >= len(view.net_nodes):
            continue
        nodes = view.net_nodes[idx]
        if use_pins and idx < len(view.net_pin_nodes):
            endpoints = _endpoint_positions(view, view.net_pin_nodes[idx], use_pins=True)
        else:
            endpoints = _endpoint_positions(view, nodes, use_pins=False)

        if len(endpoints) < 2:
            continue

        centroid = endpoints.mean(dim=0)
        degree = len(endpoints)

        # Visual emphasis for high-degree nets
        high_degree = degree >= max(10, int(view.num_macros * 0.1))
        color = "#B07AA1" if high_degree else "#9E9E9E"
        alpha = 0.45 if high_degree else 0.25
        width = 0.8 if high_degree else 0.5

        if net_style == "clique":
            for i in range(len(endpoints)):
                for j in range(i + 1, len(endpoints)):
                    ax.plot(
                        [endpoints[i, 0], endpoints[j, 0]],
                        [endpoints[i, 1], endpoints[j, 1]],
                        color=color,
                        alpha=alpha,
                        linewidth=width,
                        zorder=1,
                    )
        else:
            for pt in endpoints:
                ax.plot(
                    [pt[0].item(), centroid[0].item()],
                    [pt[1].item(), centroid[1].item()],
                    color=color,
                    alpha=alpha,
                    linewidth=width,
                    zorder=1,
                )

    ax.set_xlim(-canvas_w * 0.02, canvas_w * 1.02)
    ax.set_ylim(-canvas_h * 0.02, canvas_h * 1.02)
    ax.set_aspect("equal")
    ax.set_xlabel("X (um)")
    ax.set_ylabel("Y (um)")

    if title is None:
        title = (
            f"{view.name} | macros={view.num_macros} (hard={view.num_hard_macros}, soft={view.num_soft_macros}) "
            f"nets={view.num_nets}"
        )
    ax.set_title(title, fontsize=10)


def render_benchmark(
    ax: plt.Axes,
    data: Any,
    max_nets: Optional[int] = 300,
    net_sample: str = "largest",
    show_grid: bool = False,
    show_pins: bool = False,
    show_ports: bool = True,
    show_labels: bool = False,
    net_style: str = "star",
    seed: int = 1,
    title: Optional[str] = None,
) -> BenchmarkView:
    """Render a Benchmark-format placement onto an existing axis."""
    view = _normalize_benchmark(data)
    _render_view(
        ax=ax,
        view=view,
        max_nets=max_nets,
        net_sample=net_sample,
        show_grid=show_grid,
        show_pins=show_pins,
        show_ports=show_ports,
        show_labels=show_labels,
        net_style=net_style,
        seed=seed,
        title=title,
    )
    return view


def visualize_benchmark(
    data: Any,
    output: Optional[str] = None,
    max_nets: Optional[int] = 300,
    net_sample: str = "largest",
    show_grid: bool = False,
    show_pins: bool = False,
    show_ports: bool = True,
    show_labels: bool = False,
    net_style: str = "star",
    seed: int = 1,
    dpi: int = 150,
) -> plt.Figure:
    view = _normalize_benchmark(data)

    canvas_w = view.canvas_width
    canvas_h = view.canvas_height

    fig_w = 10.0
    fig_h = max(6.0, fig_w * (canvas_h / max(canvas_w, 1e-6)))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    _render_view(
        ax=ax,
        view=view,
        max_nets=max_nets,
        net_sample=net_sample,
        show_grid=show_grid,
        show_pins=show_pins,
        show_ports=show_ports,
        show_labels=show_labels,
        net_style=net_style,
        seed=seed,
        title=None,
    )

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=dpi, bbox_inches="tight")
        print(f"Saved visualization to {output}")

    return fig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Benchmark-format placements")
    parser.add_argument("--input", "-i", required=True, help="Input .pt file saved in Benchmark format")
    parser.add_argument("--output", "-o", default=None, help="Output image path")
    parser.add_argument("--max-nets", type=int, default=300, help="Max nets to draw")
    parser.add_argument("--net-sample", choices=["largest", "random"], default="largest")
    parser.add_argument("--net-style", choices=["star", "clique"], default="star")
    parser.add_argument("--show-grid", action="store_true", help="Draw placement grid")
    parser.add_argument("--show-pins", action="store_true", help="Draw macro pins")
    parser.add_argument("--hide-ports", action="store_true", help="Hide IO ports")
    parser.add_argument("--show-labels", action="store_true", help="Draw macro names")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for sampling nets")
    parser.add_argument("--dpi", type=int, default=150, help="Output DPI")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data = _load_any(args.input)

    output = args.output
    if output is None:
        out_name = Path(args.input).stem + "_viz.png"
        output = str(Path(args.input).with_name(out_name))

    visualize_benchmark(
        data,
        output=output,
        max_nets=args.max_nets,
        net_sample=args.net_sample,
        net_style=args.net_style,
        show_grid=args.show_grid,
        show_pins=args.show_pins,
        show_ports=not args.hide_ports,
        show_labels=args.show_labels,
        seed=args.seed,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
