"""
Synthetic benchmark generator that follows the Benchmark tensor format.

This script creates IBM-like synthetic macro placement data with:
- bimodal macro area distribution
- log-normal aspect ratios
- clustered placement (module-like regions)
- local and global nets with heavy-tailed degrees
- optional pin offsets and pin-level nets
"""

from dataclasses import dataclass, field
from typing import List, Optional
import argparse
import math
import random
from pathlib import Path

import torch


@dataclass
class Benchmark:
    """
    Placement benchmark in pure PyTorch tensors.

    All coordinates are in microns.
    All indices are 0-based.

    Tensors contain both hard macros (indices [0, num_hard_macros)) and
    soft macros (indices [num_hard_macros, num_macros)). Hard macros are
    the primary optimization targets; soft macros are standard cell clusters
    that should be co-optimized for best results.
    """

    # Core data
    name: str

    # Canvas
    canvas_width: float
    canvas_height: float

    # Macros (hard + soft, hard macros first)
    num_macros: int
    macro_positions: torch.Tensor  # [num_macros, 2] - (x, y) centers
    macro_sizes: torch.Tensor  # [num_macros, 2] - (width, height)
    macro_fixed: torch.Tensor  # [num_macros] - bool, True if fixed
    macro_names: List[str]  # [num_macros] - names for debugging

    # Nets (hypergraph connectivity)
    num_nets: int
    net_nodes: List[torch.Tensor]  # List of [nodes_in_net_i] - node indices
    net_weights: torch.Tensor  # [num_nets] - net weights (default 1.0)

    # Grid (for metrics)
    grid_rows: int
    grid_cols: int

    # I/O ports (pins on the chip boundary)
    port_positions: torch.Tensor = field(default_factory=lambda: torch.zeros(0, 2))

    # Hard macro pin offsets (relative to macro center)
    # List of [num_pins_i, 2] tensors, one per hard macro (indices [0, num_hard_macros))
    macro_pin_offsets: List[torch.Tensor] = field(default_factory=list)

    # Pin-level net connectivity (optional; empty list if not populated)
    # Each net_pin_nodes[i] is an int64 tensor of shape [num_pins_in_net_i, 2] where:
    #   column 0 = owner index:
    #     - hard macros:  [0, num_hard_macros)
    #     - soft macros:  [num_hard_macros, num_macros)
    #     - I/O ports:    [num_macros, num_macros + num_ports)
    #   column 1 = pin index within that owner:
    #     - hard macro:   index into macro_pin_offsets[owner]
    #     - soft macro:   always 0 (pins at macro center; soft macros carry no offsets)
    #     - port:         always 0 (port is a single point at port_positions[owner-num_macros])
    # Unlike net_nodes (which dedups to per-macro granularity), this preserves
    # every pin endpoint - multiple pins on the same macro appear as multiple rows.
    # Needed by placers computing pin-level HPWL for differentiable loss.
    net_pin_nodes: List[torch.Tensor] = field(default_factory=list)

    # Routing parameters
    hroutes_per_micron: float = 11.285  # Horizontal routing tracks per micron
    vroutes_per_micron: float = 12.605  # Vertical routing tracks per micron

    # PlacementCost mapping (tensor index -> PlacementCost module index)
    hard_macro_indices: List[int] = field(default_factory=list)
    soft_macro_indices: List[int] = field(default_factory=list)

    # Counts
    num_hard_macros: int = 0
    num_soft_macros: int = 0

    def __post_init__(self):
        """Validate tensor shapes and set counts."""
        if self.num_hard_macros == 0 and self.num_soft_macros == 0:
            self.num_hard_macros = self.num_macros
            self.num_soft_macros = 0

        assert self.num_macros == self.num_hard_macros + self.num_soft_macros, (
            f"num_macros {self.num_macros} != "
            f"num_hard {self.num_hard_macros} + num_soft {self.num_soft_macros}"
        )
        assert self.macro_positions.shape == (self.num_macros, 2), (
            f"macro_positions shape {self.macro_positions.shape} != ({self.num_macros}, 2)"
        )
        assert self.macro_sizes.shape == (self.num_macros, 2), (
            f"macro_sizes shape {self.macro_sizes.shape} != ({self.num_macros}, 2)"
        )
        assert self.macro_fixed.shape == (self.num_macros,), (
            f"macro_fixed shape {self.macro_fixed.shape} != ({self.num_macros},)"
        )

        if len(self.net_nodes) > 0:
            assert len(self.net_nodes) == self.num_nets, (
                f"len(net_nodes) {len(self.net_nodes)} != num_nets {self.num_nets}"
            )

        if len(self.net_pin_nodes) > 0:
            assert len(self.net_pin_nodes) == self.num_nets, (
                f"len(net_pin_nodes) {len(self.net_pin_nodes)} != num_nets {self.num_nets}"
            )

        assert self.net_weights.shape == (self.num_nets,), (
            f"net_weights shape {self.net_weights.shape} != ({self.num_nets},)"
        )

    def save(self, path: str):
        """Save benchmark to .pt file."""
        torch.save(
            {
                "name": self.name,
                "canvas_width": self.canvas_width,
                "canvas_height": self.canvas_height,
                "num_macros": self.num_macros,
                "num_hard_macros": self.num_hard_macros,
                "num_soft_macros": self.num_soft_macros,
                "macro_positions": self.macro_positions,
                "macro_sizes": self.macro_sizes,
                "macro_fixed": self.macro_fixed,
                "macro_names": self.macro_names,
                "num_nets": self.num_nets,
                "net_nodes": self.net_nodes,
                "net_weights": self.net_weights,
                "grid_rows": self.grid_rows,
                "grid_cols": self.grid_cols,
                "hroutes_per_micron": self.hroutes_per_micron,
                "vroutes_per_micron": self.vroutes_per_micron,
                "port_positions": self.port_positions,
                "macro_pin_offsets": self.macro_pin_offsets,
                "net_pin_nodes": self.net_pin_nodes,
                "hard_macro_indices": self.hard_macro_indices,
                "soft_macro_indices": self.soft_macro_indices,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "Benchmark":
        """Load benchmark from .pt file."""
        data = torch.load(path, weights_only=False)
        if "num_hard_macros" not in data:
            data["num_hard_macros"] = data["num_macros"]
            data["num_soft_macros"] = 0
        if "soft_macro_indices" not in data:
            data["soft_macro_indices"] = []
        if "port_positions" not in data:
            data["port_positions"] = torch.zeros(0, 2)
        if "macro_pin_offsets" not in data:
            data["macro_pin_offsets"] = []
        if "net_pin_nodes" not in data:
            data["net_pin_nodes"] = []
        return cls(**data)

    def get_movable_mask(self) -> torch.Tensor:
        """Return mask of movable macros (not fixed)."""
        return ~self.macro_fixed

    def get_hard_macro_mask(self) -> torch.Tensor:
        """Return mask that is True for hard macros (first num_hard_macros entries)."""
        mask = torch.zeros(self.num_macros, dtype=torch.bool)
        mask[: self.num_hard_macros] = True
        return mask

    def get_soft_macro_mask(self) -> torch.Tensor:
        """Return mask that is True for soft macros."""
        mask = torch.zeros(self.num_macros, dtype=torch.bool)
        mask[self.num_hard_macros :] = True
        return mask

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Benchmark(name='{self.name}', "
            f"hard_macros={self.num_hard_macros}, "
            f"soft_macros={self.num_soft_macros}, "
            f"num_nets={self.num_nets}, "
            f"canvas={self.canvas_width:.1f}x{self.canvas_height:.1f}um)"
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _lognormal(mean_log: float, std_log: float, count: int) -> torch.Tensor:
    if count <= 0:
        return torch.zeros(0)
    return torch.exp(torch.randn(count) * std_log + mean_log)


def _sample_areas(
    num_hard: int,
    num_soft: int,
    hard_large_frac: float,
    min_area: float,
) -> torch.Tensor:
    num_large = int(round(num_hard * hard_large_frac))
    num_small = max(0, num_hard - num_large)

    small = _lognormal(math.log(12.0), 0.55, num_small)
    large = _lognormal(math.log(180.0), 0.60, num_large)

    if num_hard > 0:
        hard = torch.cat([small, large], dim=0)
        hard = hard[torch.randperm(num_hard)]
    else:
        hard = torch.zeros(0)

    soft = _lognormal(math.log(4.0), 0.45, num_soft)

    areas = torch.cat([hard, soft], dim=0)
    return areas.clamp(min=min_area)


def _sample_aspect_ratios(num_macros: int, aspect_sigma: float) -> torch.Tensor:
    return torch.exp(torch.randn(num_macros) * aspect_sigma)


def _build_canvas(total_area: float, utilization: float, aspect: float) -> tuple[float, float]:
    canvas_area = total_area / max(utilization, 1e-6)
    width = math.sqrt(canvas_area * aspect)
    height = canvas_area / width
    return width, height


def _assign_modules(num_macros: int, num_modules: int) -> torch.Tensor:
    if num_modules <= 1:
        return torch.zeros(num_macros, dtype=torch.long)

    raw = torch.rand(num_modules).pow(1.7)
    weights = raw / raw.sum()
    return torch.multinomial(weights, num_macros, replacement=True)


def _sample_positions(
    widths: torch.Tensor,
    heights: torch.Tensor,
    canvas_w: float,
    canvas_h: float,
    module_ids: torch.Tensor,
    num_modules: int,
    cluster_spread: float,
) -> torch.Tensor:
    centers = torch.zeros(num_modules, 2)
    centers[:, 0] = torch.rand(num_modules) * canvas_w
    centers[:, 1] = torch.rand(num_modules) * canvas_h

    sigma = min(canvas_w, canvas_h) * cluster_spread
    positions = torch.zeros(len(widths), 2)

    for i in range(len(widths)):
        m = int(module_ids[i].item())
        pos = centers[m] + torch.randn(2) * sigma

        w = float(widths[i].item())
        h = float(heights[i].item())

        x = float(pos[0].item())
        y = float(pos[1].item())

        x = min(max(x, w / 2.0), canvas_w - w / 2.0)
        y = min(max(y, h / 2.0), canvas_h - h / 2.0)

        positions[i, 0] = x
        positions[i, 1] = y

    return positions


def _place_fixed_macros(
    positions: torch.Tensor,
    widths: torch.Tensor,
    heights: torch.Tensor,
    fixed_indices: List[int],
    canvas_w: float,
    canvas_h: float,
    edge_band: float,
) -> None:
    for idx in fixed_indices:
        w = float(widths[idx].item())
        h = float(heights[idx].item())
        side = random.choice(["left", "right", "bottom", "top"])

        if side in ("left", "right"):
            x = w / 2.0 + edge_band if side == "left" else canvas_w - w / 2.0 - edge_band
            y = random.uniform(h / 2.0, canvas_h - h / 2.0)
        else:
            y = h / 2.0 + edge_band if side == "bottom" else canvas_h - h / 2.0 - edge_band
            x = random.uniform(w / 2.0, canvas_w - w / 2.0)

        positions[idx, 0] = x
        positions[idx, 1] = y


def _sample_port_positions(num_ports: int, canvas_w: float, canvas_h: float) -> torch.Tensor:
    if num_ports <= 0:
        return torch.zeros(0, 2)

    positions = torch.zeros(num_ports, 2)
    for i in range(num_ports):
        side = random.choice(["left", "right", "bottom", "top"])
        if side == "left":
            x = 0.0
            y = random.uniform(0.0, canvas_h)
        elif side == "right":
            x = canvas_w
            y = random.uniform(0.0, canvas_h)
        elif side == "bottom":
            x = random.uniform(0.0, canvas_w)
            y = 0.0
        else:
            x = random.uniform(0.0, canvas_w)
            y = canvas_h
        positions[i, 0] = x
        positions[i, 1] = y

    return positions


def _sample_pin_offsets(
    widths: torch.Tensor,
    heights: torch.Tensor,
    num_hard: int,
    min_pins: int,
    max_pins: int,
) -> List[torch.Tensor]:
    offsets: List[torch.Tensor] = []

    for i in range(num_hard):
        w = float(widths[i].item())
        h = float(heights[i].item())
        num_pins = max(min_pins, min(max_pins, random.randint(min_pins, max_pins)))

        pins = torch.zeros(num_pins, 2)
        for p in range(num_pins):
            side = random.choice(["left", "right", "bottom", "top"])
            if side == "left":
                x = -w / 2.0
                y = random.uniform(-h / 2.0, h / 2.0)
            elif side == "right":
                x = w / 2.0
                y = random.uniform(-h / 2.0, h / 2.0)
            elif side == "bottom":
                x = random.uniform(-w / 2.0, w / 2.0)
                y = -h / 2.0
            else:
                x = random.uniform(-w / 2.0, w / 2.0)
                y = h / 2.0
            pins[p, 0] = x
            pins[p, 1] = y

        offsets.append(pins)

    return offsets


def _sample_degrees(num_nets: int, max_degree: int, alpha: float) -> torch.Tensor:
    max_degree = max(2, max_degree)
    degrees = torch.arange(2, max_degree + 1, dtype=torch.float32)
    weights = degrees.pow(-alpha)
    weights = weights / weights.sum()
    idx = torch.multinomial(weights, num_nets, replacement=True)
    return degrees[idx].to(dtype=torch.long)


def _weighted_sample_candidates(
    candidates: torch.Tensor,
    weights: torch.Tensor,
    k: int,
) -> torch.Tensor:
    if len(candidates) == 0:
        return torch.multinomial(weights, k, replacement=False)

    if len(candidates) >= k:
        local_weights = weights[candidates]
        local_weights = local_weights / local_weights.sum()
        picked = torch.multinomial(local_weights, k, replacement=False)
        return candidates[picked]

    remaining = k - len(candidates)
    mask = torch.ones(len(weights), dtype=torch.bool)
    mask[candidates] = False

    rest_indices = torch.arange(len(weights))[mask]
    if remaining > len(rest_indices):
        remaining = len(rest_indices)

    rest_weights = weights[rest_indices]
    rest_weights = rest_weights / rest_weights.sum()
    extra = torch.multinomial(rest_weights, remaining, replacement=False)

    return torch.cat([candidates, rest_indices[extra]], dim=0)


def _build_nets(
    num_nets: int,
    num_macros: int,
    degrees: torch.Tensor,
    areas: torch.Tensor,
    module_ids: torch.Tensor,
    num_modules: int,
    local_prob: float,
) -> List[torch.Tensor]:
    area_weights = areas.clamp(min=1e-6).pow(0.3)
    area_weights = area_weights / area_weights.sum()

    module_sizes = torch.zeros(num_modules, dtype=torch.float32)
    for m in range(num_modules):
        module_sizes[m] = float((module_ids == m).sum().item())
    module_weights = module_sizes / module_sizes.sum()

    nets: List[torch.Tensor] = []

    for i in range(num_nets):
        deg = int(degrees[i].item())
        deg = min(max(deg, 2), num_macros)

        if num_modules > 1 and random.random() < local_prob:
            module = int(torch.multinomial(module_weights, 1).item())
            candidates = torch.nonzero(module_ids == module, as_tuple=False).flatten()
            chosen = _weighted_sample_candidates(candidates, area_weights, deg)
        else:
            chosen = torch.multinomial(area_weights, deg, replacement=False)

        chosen = torch.unique(chosen, sorted=True)
        if len(chosen) < 2:
            chosen = torch.multinomial(area_weights, 2, replacement=False)
            chosen = torch.unique(chosen, sorted=True)

        nets.append(chosen.to(dtype=torch.long))

    return nets


def _build_pin_nets(
    net_nodes: List[torch.Tensor],
    num_hard: int,
    macro_pin_offsets: List[torch.Tensor],
    num_macros: int,
    num_ports: int,
    port_net_frac: float,
) -> List[torch.Tensor]:
    net_pin_nodes: List[torch.Tensor] = []

    for nodes in net_nodes:
        rows = []
        for macro_idx in nodes.tolist():
            if macro_idx < num_hard:
                pin_count = macro_pin_offsets[macro_idx].shape[0]
                pin_idx = random.randrange(pin_count) if pin_count > 0 else 0
            else:
                pin_idx = 0
            rows.append([macro_idx, pin_idx])

        if num_ports > 0 and random.random() < port_net_frac:
            port_idx = random.randrange(num_ports)
            rows.append([num_macros + port_idx, 0])

        net_pin_nodes.append(torch.tensor(rows, dtype=torch.long))

    return net_pin_nodes


def generate_benchmark(
    name: str,
    num_macros: int,
    num_hard_macros: int,
    num_soft_macros: int,
    num_fixed_macros: int,
    num_nets: int,
    num_ports: int,
    grid_rows: int,
    grid_cols: int,
    utilization: Optional[float],
    canvas_aspect: Optional[float],
    hard_large_frac: float,
    aspect_sigma: float,
    cluster_spread: float,
    local_net_prob: float,
    max_degree: int,
    degree_alpha: float,
    net_weight_std: float,
    min_pins: int,
    max_pins: int,
    emit_pin_nets: bool,
    port_net_frac: float,
) -> Benchmark:
    if num_macros < 2:
        raise ValueError("num_macros must be at least 2")
    if num_hard_macros < 0 or num_soft_macros < 0:
        raise ValueError("num_hard_macros and num_soft_macros must be non-negative")
    if num_fixed_macros > num_hard_macros:
        raise ValueError("num_fixed_macros must be <= num_hard_macros")
    if num_nets < 1:
        raise ValueError("num_nets must be at least 1")

    areas = _sample_areas(num_hard_macros, num_soft_macros, hard_large_frac, min_area=1.0)
    aspects = _sample_aspect_ratios(num_macros, aspect_sigma)
    widths = torch.sqrt(areas * aspects)
    heights = areas / widths

    if utilization is None:
        utilization = float(torch.empty(1).uniform_(0.35, 0.65).item())
    if canvas_aspect is None:
        canvas_aspect = float(torch.empty(1).uniform_(0.8, 1.35).item())

    canvas_w, canvas_h = _build_canvas(float(areas.sum().item()), utilization, canvas_aspect)

    num_modules = max(2, min(8, int(math.sqrt(num_macros)) + 1))
    module_ids = _assign_modules(num_macros, num_modules)
    positions = _sample_positions(
        widths,
        heights,
        canvas_w,
        canvas_h,
        module_ids,
        num_modules,
        cluster_spread,
    )

    fixed_indices = []
    if num_fixed_macros > 0:
        fixed_indices = random.sample(range(num_hard_macros), num_fixed_macros)
        edge_band = min(canvas_w, canvas_h) * 0.03
        _place_fixed_macros(
            positions,
            widths,
            heights,
            fixed_indices,
            canvas_w,
            canvas_h,
            edge_band,
        )

    macro_fixed = torch.zeros(num_macros, dtype=torch.bool)
    if fixed_indices:
        macro_fixed[torch.tensor(fixed_indices, dtype=torch.long)] = True

    macro_positions = positions.to(dtype=torch.float32)
    macro_sizes = torch.stack([widths, heights], dim=1).to(dtype=torch.float32)

    hard_macro_indices = list(range(num_hard_macros))
    soft_macro_indices = list(range(num_hard_macros, num_macros))

    macro_names = []
    for i in range(num_macros):
        if i < num_hard_macros:
            macro_names.append(f"H{i:04d}")
        else:
            macro_names.append(f"S{i - num_hard_macros:04d}")

    degrees = _sample_degrees(num_nets, min(max_degree, num_macros), degree_alpha)
    net_nodes = _build_nets(
        num_nets,
        num_macros,
        degrees,
        areas,
        module_ids,
        num_modules,
        local_net_prob,
    )

    net_weights = torch.exp(torch.randn(num_nets) * net_weight_std).clamp(min=0.1, max=10.0)
    net_weights = net_weights.to(dtype=torch.float32)

    port_positions = _sample_port_positions(num_ports, canvas_w, canvas_h)

    macro_pin_offsets = _sample_pin_offsets(widths, heights, num_hard_macros, min_pins, max_pins)

    if emit_pin_nets:
        net_pin_nodes = _build_pin_nets(
            net_nodes,
            num_hard_macros,
            macro_pin_offsets,
            num_macros,
            num_ports,
            port_net_frac,
        )
    else:
        net_pin_nodes = []

    benchmark = Benchmark(
        name=name,
        canvas_width=float(canvas_w),
        canvas_height=float(canvas_h),
        num_macros=num_macros,
        macro_positions=macro_positions,
        macro_sizes=macro_sizes,
        macro_fixed=macro_fixed,
        macro_names=macro_names,
        num_nets=num_nets,
        net_nodes=net_nodes,
        net_weights=net_weights,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        port_positions=port_positions,
        macro_pin_offsets=macro_pin_offsets,
        net_pin_nodes=net_pin_nodes,
        hard_macro_indices=hard_macro_indices,
        soft_macro_indices=soft_macro_indices,
        num_hard_macros=num_hard_macros,
        num_soft_macros=num_soft_macros,
    )

    return benchmark


def _resolve_counts(
    num_macros: Optional[int],
    num_hard: Optional[int],
    num_soft: Optional[int],
) -> tuple[int, int, int]:
    if num_macros is None:
        if num_hard is None or num_soft is None:
            raise ValueError("Provide num_macros or both num_hard_macros and num_soft_macros")
        return num_hard + num_soft, num_hard, num_soft

    if num_hard is None and num_soft is None:
        return num_macros, num_macros, 0

    if num_hard is None:
        if num_soft is None:
            return num_macros, num_macros, 0
        return num_macros, num_macros - num_soft, num_soft

    if num_soft is None:
        return num_macros, num_hard, num_macros - num_hard

    if num_hard + num_soft != num_macros:
        raise ValueError("num_hard_macros + num_soft_macros must equal num_macros")

    return num_macros, num_hard, num_soft


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic macro placement benchmark")

    parser.add_argument("--out", type=str, required=True, help="Output .pt path")
    parser.add_argument("--name", type=str, default="synthetic_ibm_like", help="Benchmark name")

    parser.add_argument("--num-macros", type=int, default=200, help="Total macros")
    parser.add_argument("--num-hard-macros", type=int, default=None, help="Hard macro count")
    parser.add_argument("--num-soft-macros", type=int, default=None, help="Soft macro count")
    parser.add_argument("--num-fixed-macros", type=int, default=10, help="Fixed hard macros")

    parser.add_argument("--num-nets", type=int, default=None, help="Total nets")
    parser.add_argument("--net-ratio", type=float, default=1.4, help="Nets per macro if num-nets is unset")

    parser.add_argument("--num-ports", type=int, default=0, help="Number of IO ports")
    parser.add_argument("--port-net-frac", type=float, default=0.0, help="Fraction of nets with a port pin")

    parser.add_argument("--grid-rows", type=int, default=64, help="Grid rows")
    parser.add_argument("--grid-cols", type=int, default=64, help="Grid cols")

    parser.add_argument("--utilization", type=float, default=None, help="Utilization target (0-1)")
    parser.add_argument("--canvas-aspect", type=float, default=None, help="Canvas aspect ratio (w/h)")

    parser.add_argument("--hard-large-frac", type=float, default=0.25, help="Fraction of large hard macros")
    parser.add_argument("--aspect-sigma", type=float, default=0.55, help="Log-normal aspect ratio sigma")
    parser.add_argument("--cluster-spread", type=float, default=0.12, help="Cluster spread as canvas fraction")

    parser.add_argument("--local-net-prob", type=float, default=0.75, help="Probability of local nets")
    parser.add_argument("--max-degree", type=int, default=20, help="Max net degree")
    parser.add_argument("--degree-alpha", type=float, default=1.15, help="Zipf alpha for net degrees")
    parser.add_argument("--net-weight-std", type=float, default=0.35, help="Stddev for log-normal net weights")

    parser.add_argument("--min-pins", type=int, default=1, help="Min pins per hard macro")
    parser.add_argument("--max-pins", type=int, default=4, help="Max pins per hard macro")
    parser.add_argument("--emit-pin-nets", action="store_true", help="Emit net_pin_nodes with pin indices")

    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument(
        "--export-plc-dir",
        type=str,
        default=None,
        help="Optional output directory to also write netlist.pb.txt and initial.plc",
    )
    parser.add_argument(
        "--plc-design-name",
        type=str,
        default=None,
        help="Optional block/design name used in exported initial.plc header",
    )
    parser.add_argument(
        "--plc-coarse-nets",
        action="store_true",
        help="Use coarse net_nodes instead of pin-level net_pin_nodes when exporting PLC",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    num_macros, num_hard, num_soft = _resolve_counts(
        args.num_macros,
        args.num_hard_macros,
        args.num_soft_macros,
    )

    if args.num_nets is None:
        num_nets = max(1, int(round(num_macros * args.net_ratio)))
    else:
        num_nets = args.num_nets

    benchmark = generate_benchmark(
        name=args.name,
        num_macros=num_macros,
        num_hard_macros=num_hard,
        num_soft_macros=num_soft,
        num_fixed_macros=args.num_fixed_macros,
        num_nets=num_nets,
        num_ports=args.num_ports,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        utilization=args.utilization,
        canvas_aspect=args.canvas_aspect,
        hard_large_frac=args.hard_large_frac,
        aspect_sigma=args.aspect_sigma,
        cluster_spread=args.cluster_spread,
        local_net_prob=args.local_net_prob,
        max_degree=args.max_degree,
        degree_alpha=args.degree_alpha,
        net_weight_std=args.net_weight_std,
        min_pins=args.min_pins,
        max_pins=args.max_pins,
        emit_pin_nets=args.emit_pin_nets,
        port_net_frac=args.port_net_frac,
    )

    benchmark.save(args.out)

    if args.export_plc_dir:
        try:
            from .dreamplace_bridge import export_plc
        except Exception:
            from dreamplace_bridge import export_plc  # type: ignore[no-redef]

        export_result = export_plc(
            benchmark_path=args.out,
            out_dir=args.export_plc_dir,
            design_name=args.plc_design_name or Path(args.out).stem,
            use_pin_nets=not args.plc_coarse_nets,
        )
        print(f"Exported PlacementCost files to {export_result['out_dir']}")
        print(f"  netlist: {export_result['netlist']}")
        print(f"  plc: {export_result['plc']}")


if __name__ == "__main__":
    main()
