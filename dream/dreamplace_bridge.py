"""
Bridge between Benchmark tensor format and DREAMPlace Bookshelf format.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

try:
    from .generate_benchmark import Benchmark
except Exception:
    try:
        from generate_benchmark import Benchmark  # type: ignore[assignment]
    except Exception:
        Benchmark = None  # type: ignore[assignment]


def _get_field(data: Any, key: str, default: Any = None) -> Any:
    if isinstance(data, dict):
        return data.get(key, default)
    return getattr(data, key, default)


def _coerce_tensor(value: Any, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype)


def _coerce_list_of_tensors(items: Any, dtype: torch.dtype) -> List[torch.Tensor]:
    if items is None:
        return []
    out: List[torch.Tensor] = []
    for item in items:
        out.append(torch.as_tensor(item, dtype=dtype))
    return out


def load_benchmark_any(path: str) -> Any:
    return torch.load(path)


def save_benchmark_any(data: Any, path: str) -> None:
    if Benchmark is not None and isinstance(data, Benchmark):
        data.save(path)
    else:
        torch.save(data, path)


def _normalize_benchmark(data: Any) -> Dict[str, Any]:
    num_macros = int(_get_field(data, "num_macros"))
    num_hard = int(_get_field(data, "num_hard_macros", num_macros))
    num_soft = int(_get_field(data, "num_soft_macros", num_macros - num_hard))

    macro_positions = _coerce_tensor(_get_field(data, "macro_positions"), torch.float32)
    macro_sizes = _coerce_tensor(_get_field(data, "macro_sizes"), torch.float32)
    macro_fixed = _coerce_tensor(_get_field(data, "macro_fixed"), torch.bool)

    macro_names = _get_field(data, "macro_names", None)
    if macro_names is None:
        macro_names = [f"M{i:04d}" for i in range(num_macros)]

    net_nodes = _coerce_list_of_tensors(_get_field(data, "net_nodes", []), torch.long)
    net_pin_nodes = _coerce_list_of_tensors(
        _get_field(data, "net_pin_nodes", []), torch.long
    )
    net_weights = _coerce_tensor(
        _get_field(data, "net_weights", torch.ones(len(net_nodes))), torch.float32
    )

    port_positions = _coerce_tensor(
        _get_field(data, "port_positions", torch.zeros(0, 2)), torch.float32
    )
    macro_pin_offsets = _coerce_list_of_tensors(
        _get_field(data, "macro_pin_offsets", []), torch.float32
    )

    return {
        "name": str(_get_field(data, "name", "benchmark")),
        "canvas_width": float(
            _get_field(data, "canvas_width", float(macro_positions[:, 0].max().item()))
        ),
        "canvas_height": float(
            _get_field(data, "canvas_height", float(macro_positions[:, 1].max().item()))
        ),
        "num_macros": num_macros,
        "num_hard_macros": num_hard,
        "num_soft_macros": num_soft,
        "macro_positions": macro_positions,
        "macro_sizes": macro_sizes,
        "macro_fixed": macro_fixed,
        "macro_names": list(macro_names),
        "num_nets": int(_get_field(data, "num_nets", len(net_nodes))),
        "net_nodes": net_nodes,
        "net_pin_nodes": net_pin_nodes,
        "net_weights": net_weights,
        "grid_rows": int(_get_field(data, "grid_rows", 0)),
        "grid_cols": int(_get_field(data, "grid_cols", 0)),
        "hroutes_per_micron": float(_get_field(data, "hroutes_per_micron", 11.285)),
        "vroutes_per_micron": float(_get_field(data, "vroutes_per_micron", 12.605)),
        "port_positions": port_positions,
        "macro_pin_offsets": macro_pin_offsets,
    }


def _sanitize_name(name: str, used: set[str]) -> str:
    base = "".join(c if (c.isalnum() or c in "_.-") else "_" for c in name)
    if not base:
        base = "node"
    candidate = base
    idx = 1
    while candidate in used:
        candidate = f"{base}_{idx}"
        idx += 1
    used.add(candidate)
    return candidate


def _scale_int(value: float, scale: float, min_value: int = 0) -> int:
    return max(min_value, int(round(value * scale)))


def _write_aux(aux_path: Path, design_name: str) -> None:
    with aux_path.open("w", encoding="utf-8") as f:
        f.write(
            f"RowBasedPlacement : {design_name}.nodes {design_name}.nets "
            f"{design_name}.pl {design_name}.scl"
        )


def _write_nodes(
    nodes_path: Path,
    node_names: List[str],
    widths: List[int],
    heights: List[int],
    terminal_flags: List[str],
) -> None:
    num_nodes = len(node_names)
    num_terminals = sum(1 for flag in terminal_flags if flag)

    with nodes_path.open("w", encoding="utf-8") as f:
        f.write("UCLA nodes 1.0\n\n")
        f.write(f"NumNodes : {num_nodes}\n")
        f.write(f"NumTerminals : {num_terminals}\n\n")
        for name, w, h, flag in zip(node_names, widths, heights, terminal_flags):
            if flag:
                f.write(f"{name} {w} {h} {flag}\n")
            else:
                f.write(f"{name} {w} {h}\n")


def _write_pl(
    pl_path: Path,
    node_names: List[str],
    x_ll: List[int],
    y_ll: List[int],
    fixed_flags: List[str],
) -> None:
    with pl_path.open("w", encoding="utf-8") as f:
        f.write("UCLA pl 1.0\n\n")
        for name, x, y, flag in zip(node_names, x_ll, y_ll, fixed_flags):
            line = f"{name} {x} {y} : N"
            if flag:
                line += f" {flag}"
            f.write(line + "\n")


def _write_nets(
    nets_path: Path,
    nets: List[List[Tuple[str, int, int]]],
) -> None:
    num_nets = len(nets)
    num_pins = sum(len(pins) for pins in nets)

    with nets_path.open("w", encoding="utf-8") as f:
        f.write("UCLA nets 1.0\n\n")
        f.write(f"NumNets : {num_nets}\n")
        f.write(f"NumPins : {num_pins}\n\n")
        for idx, pins in enumerate(nets):
            f.write(f"NetDegree : {len(pins)} N{idx}\n")
            for name, dx, dy in pins:
                f.write(f"    {name} I : {dx} {dy}\n")


def _write_scl(
    scl_path: Path,
    canvas_w: int,
    canvas_h: int,
    row_height: int,
    site_width: int,
) -> None:
    num_rows = max(1, int(math.ceil(canvas_h / max(row_height, 1))))
    num_sites = max(1, int(math.ceil(canvas_w / max(site_width, 1))))

    with scl_path.open("w", encoding="utf-8") as f:
        f.write("UCLA scl 1.0\n\n")
        f.write(f"NumRows : {num_rows}\n\n")
        for r in range(num_rows):
            y = r * row_height
            if y >= canvas_h:
                break
            f.write("CoreRow Horizontal\n")
            f.write(f"  Coordinate    :   {y}\n")
            f.write(f"  Height        :   {row_height}\n")
            f.write(f"  Sitewidth     :    {site_width}\n")
            f.write(f"  Sitespacing   :    {site_width}\n")
            f.write("  Siteorient    :    1\n")
            f.write("  Sitesymmetry  :    1\n")
            f.write(f"  SubrowOrigin  :    0\tNumSites  :  {num_sites}\n")
            f.write("End\n")


def _build_net_pins(
    benchmark: Dict[str, Any],
    macro_names: List[str],
    port_names: List[str],
    scale: float,
    include_pin_offsets: bool,
) -> List[List[Tuple[str, int, int]]]:
    net_nodes = benchmark["net_nodes"]
    net_pin_nodes = benchmark["net_pin_nodes"]
    macro_pin_offsets = benchmark["macro_pin_offsets"]
    num_macros = benchmark["num_macros"]
    num_hard = benchmark["num_hard_macros"]
    num_ports = len(port_names)

    nets: List[List[Tuple[str, int, int]]] = []

    use_pin_nodes = (
        include_pin_offsets
        and len(net_pin_nodes) == len(net_nodes)
        and len(net_pin_nodes) > 0
    )

    for i, nodes in enumerate(net_nodes):
        pins: List[Tuple[str, int, int]] = []
        if use_pin_nodes:
            pin_rows = net_pin_nodes[i]
            for owner_idx, pin_idx in pin_rows.tolist():
                if owner_idx < num_macros:
                    name = macro_names[owner_idx]
                    dx = 0
                    dy = 0
                    if owner_idx < num_hard and owner_idx < len(macro_pin_offsets):
                        offsets = macro_pin_offsets[owner_idx]
                        if 0 <= pin_idx < len(offsets):
                            dx = _scale_int(float(offsets[pin_idx, 0].item()), scale)
                            dy = _scale_int(float(offsets[pin_idx, 1].item()), scale)
                    pins.append((name, dx, dy))
                else:
                    port_idx = owner_idx - num_macros
                    if 0 <= port_idx < num_ports:
                        name = port_names[port_idx]
                        pins.append((name, 0, 0))
        else:
            for owner_idx in nodes.tolist():
                if 0 <= owner_idx < num_macros:
                    name = macro_names[owner_idx]
                    pins.append((name, 0, 0))

        nets.append(pins)

    return nets


def export_bookshelf(
    benchmark_path: str,
    out_dir: str,
    design_name: Optional[str] = None,
    scale: float = 1000.0,
    row_height_um: float = 1.0,
    site_width_um: float = 1.0,
    include_ports: bool = True,
    include_pin_offsets: bool = True,
    port_size_um: float = 1.0,
) -> Dict[str, Any]:
    data = load_benchmark_any(benchmark_path)
    bench = _normalize_benchmark(data)

    if design_name is None:
        design_name = Path(benchmark_path).stem

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    used_names: set[str] = set()
    macro_names = [_sanitize_name(name, used_names) for name in bench["macro_names"]]

    num_ports = int(bench["port_positions"].shape[0]) if include_ports else 0
    port_names = []
    for idx in range(num_ports):
        port_names.append(_sanitize_name(f"P{idx:04d}", used_names))

    macro_sizes = bench["macro_sizes"]
    macro_positions = bench["macro_positions"]
    macro_fixed = bench["macro_fixed"]

    widths = [
        _scale_int(float(macro_sizes[i, 0].item()), scale, min_value=1)
        for i in range(bench["num_macros"])
    ]
    heights = [
        _scale_int(float(macro_sizes[i, 1].item()), scale, min_value=1)
        for i in range(bench["num_macros"])
    ]

    port_width = _scale_int(port_size_um, scale, min_value=1)
    port_height = _scale_int(port_size_um, scale, min_value=1)

    node_names = list(macro_names) + list(port_names)
    node_widths = list(widths) + [port_width] * num_ports
    node_heights = list(heights) + [port_height] * num_ports

    terminal_flags: List[str] = []
    fixed_flags: List[str] = []
    for i in range(bench["num_macros"]):
        if bool(macro_fixed[i].item()):
            terminal_flags.append("terminal")
            fixed_flags.append("/FIXED")
        else:
            terminal_flags.append("")
            fixed_flags.append("")

    for _ in range(num_ports):
        terminal_flags.append("terminal_NI")
        fixed_flags.append("/FIXED_NI")

    x_ll: List[int] = []
    y_ll: List[int] = []
    for i in range(bench["num_macros"]):
        w = widths[i]
        h = heights[i]
        x = _scale_int(float(macro_positions[i, 0].item()), scale) - int(round(w / 2.0))
        y = _scale_int(float(macro_positions[i, 1].item()), scale) - int(round(h / 2.0))
        x_ll.append(x)
        y_ll.append(y)

    if include_ports and num_ports > 0:
        port_positions = bench["port_positions"]
        for i in range(num_ports):
            x = _scale_int(float(port_positions[i, 0].item()), scale)
            y = _scale_int(float(port_positions[i, 1].item()), scale)
            x_ll.append(x)
            y_ll.append(y)

    nets = _build_net_pins(bench, macro_names, port_names, scale, include_pin_offsets)

    canvas_w = _scale_int(bench["canvas_width"], scale, min_value=1)
    canvas_h = _scale_int(bench["canvas_height"], scale, min_value=1)
    row_height = _scale_int(row_height_um, scale, min_value=1)
    site_width = _scale_int(site_width_um, scale, min_value=1)

    aux_path = out_path / f"{design_name}.aux"
    nodes_path = out_path / f"{design_name}.nodes"
    nets_path = out_path / f"{design_name}.nets"
    pl_path = out_path / f"{design_name}.pl"
    scl_path = out_path / f"{design_name}.scl"

    _write_aux(aux_path, design_name)
    _write_nodes(nodes_path, node_names, node_widths, node_heights, terminal_flags)
    _write_nets(nets_path, nets)
    _write_pl(pl_path, node_names, x_ll, y_ll, fixed_flags)
    _write_scl(scl_path, canvas_w, canvas_h, row_height, site_width)

    mapping = {
        "design_name": design_name,
        "scale": scale,
        "macro_names": macro_names,
        "port_names": port_names,
        "macro_count": bench["num_macros"],
        "port_count": num_ports,
        "canvas_width": bench["canvas_width"],
        "canvas_height": bench["canvas_height"],
    }

    mapping_path = out_path / f"{design_name}.map.json"
    with mapping_path.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)

    return {
        "aux": str(aux_path),
        "nodes": str(nodes_path),
        "nets": str(nets_path),
        "pl": str(pl_path),
        "scl": str(scl_path),
        "mapping": str(mapping_path),
        "design_name": design_name,
    }


def _parse_pl(pl_path: str) -> Dict[str, Tuple[int, int, bool]]:
    positions: Dict[str, Tuple[int, int, bool]] = {}
    with open(pl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            if len(tokens) < 4:
                continue
            name = tokens[0]
            try:
                x = int(float(tokens[1]))
                y = int(float(tokens[2]))
            except ValueError:
                continue
            fixed = any(tok.upper().startswith("/FIXED") for tok in tokens)
            positions[name] = (x, y, fixed)
    return positions


def import_bookshelf_solution(
    original_benchmark_path: str,
    pl_path: str,
    mapping_path: str,
    output_path: str,
    update_fixed_from_pl: bool = False,
    name_suffix: str = "_dreamplace",
) -> None:
    original = load_benchmark_any(original_benchmark_path)
    print(f"Loaded original benchmark from {original_benchmark_path}")
    bench = _normalize_benchmark(original)

    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    scale = float(mapping["scale"])
    macro_names = mapping["macro_names"]
    port_names = mapping.get("port_names", [])

    name_to_macro = {name: idx for idx, name in enumerate(macro_names)}
    name_to_port = {name: idx for idx, name in enumerate(port_names)}

    positions = _parse_pl(pl_path)

    macro_positions = bench["macro_positions"].clone()
    macro_sizes = bench["macro_sizes"]
    macro_fixed = bench["macro_fixed"].clone()

    for name, (x_ll, y_ll, fixed) in positions.items():
        if name in name_to_macro:
            idx = name_to_macro[name]
            w = float(macro_sizes[idx, 0].item())
            h = float(macro_sizes[idx, 1].item())
            cx = x_ll / scale + w / 2.0
            cy = y_ll / scale + h / 2.0
            macro_positions[idx, 0] = cx
            macro_positions[idx, 1] = cy
            if update_fixed_from_pl:
                macro_fixed[idx] = bool(fixed)

    port_positions = bench["port_positions"].clone()
    for name, (x_ll, y_ll, _fixed) in positions.items():
        if name in name_to_port:
            idx = name_to_port[name]
            port_positions[idx, 0] = x_ll / scale
            port_positions[idx, 1] = y_ll / scale

    updated = Benchmark(
        name=str(bench["name"]) + name_suffix,
        canvas_width=bench["canvas_width"],
        canvas_height=bench["canvas_height"],
        num_macros=bench["num_macros"],
        macro_positions=macro_positions,
        macro_sizes=bench["macro_sizes"],
        macro_fixed=macro_fixed,
        macro_names=bench["macro_names"],
        num_nets=bench["num_nets"],
        net_nodes=bench["net_nodes"],
        net_weights=bench["net_weights"],
        grid_rows=bench["grid_rows"],
        grid_cols=bench["grid_cols"],
        port_positions=port_positions,
        macro_pin_offsets=bench["macro_pin_offsets"],
        net_pin_nodes=bench["net_pin_nodes"],
        num_hard_macros=bench["num_hard_macros"],
        num_soft_macros=bench["num_soft_macros"],
        hard_macro_indices=_get_field(original, "hard_macro_indices", []),
        soft_macro_indices=_get_field(original, "soft_macro_indices", []),
    )

    save_benchmark_any(updated, output_path)
    return macro_positions


def write_dreamplace_config(
    config_path: str,
    aux_path: str,
    canvas_w: float,
    canvas_h: float,
    grid_rows: int,
    grid_cols: int,
    num_bins_x: Optional[int] = None,
    num_bins_y: Optional[int] = None,
    iterations: int = 200,
    learning_rate: float = 0.01,
    target_density: float = 0.8,
    gpu: int = 0,
    seed: int = 123,
    result_dir: Optional[str] = None,
    macro_place_flag: int = 0,
) -> Dict[str, Any]:
    if num_bins_x is None:
        if grid_cols > 0:
            num_bins_x = grid_cols
        else:
            num_bins_x = int(max(8, min(512, round(canvas_w / 10.0))))
    if num_bins_y is None:
        if grid_rows > 0:
            num_bins_y = grid_rows
        else:
            num_bins_y = int(max(8, min(512, round(canvas_h / 10.0))))

    if result_dir is None:
        result_dir = str(Path(config_path).parent / "results")

    # config = {
    #     "aux_input": os.path.abspath(aux_path),
    #     "gpu": gpu,
    #     "num_bins_x": int(num_bins_x),
    #     "num_bins_y": int(num_bins_y),
    #     "global_place_stages": [
    #         {
    #             "num_bins_x": int(num_bins_x),
    #             "num_bins_y": int(num_bins_y),
    #             "iteration": int(iterations),
    #             "learning_rate": float(learning_rate),
    #             "wirelength": "weighted_average",
    #             "optimizer": "nesterov",
    #             "Llambda_density_weight_iteration": 1,
    #             "Lsub_iteration": 1,
    #         }
    #     ],
    #     "target_density": float(target_density),
    #     "density_weight": 8e-5,
    #     "gamma": 4.0,
    #     "random_seed": int(seed),
    #     "scale_factor": 1.0,
    #     "ignore_net_degree": 100,
    #     "enable_fillers": 1,
    #     "gp_noise_ratio": 0.025,
    #     "global_place_flag": 1,
    #     "legalize_flag": 1,
    #     "detailed_place_flag": 0,
    #     "plot_flag": 0,
    #     "result_dir": result_dir,
    #     "macro_place_flag": int(macro_place_flag),
    # }
    config = {
        "aux_input": os.path.abspath(aux_path),
        "result_dir": result_dir,
        "gpu": 1,
        "num_bins_x": 512,
        "num_bins_y": 512,
        "global_place_stages": [
            {
                "num_bins_x": 512,
                "num_bins_y": 512,
                "iteration": 1000,
                "learning_rate": 0.01,
                "wirelength": "weighted_average",
                "optimizer": "nesterov",
                "Llambda_density_weight_iteration": 1,
                "Lsub_iteration": 1,
            }
        ],
        "target_density": 1.0,
        "density_weight": 8e-5,
        "gamma": 4.0,
        "random_seed": 1000,
        "scale_factor": 1.0,
        "ignore_net_degree": 100,
        "enable_fillers": 1,
        "gp_noise_ratio": 0.025,
        "global_place_flag": 1,
        "legalize_flag": 1,
        "detailed_place_flag": 1,
        "detailed_place_engine": "",
        "detailed_place_command": "",
        "stop_overflow": 0.07,
        "dtype": "float32",
        "plot_flag": 0,
        "random_center_init_flag": 1,
        "sort_nets_by_degree": 0,
        "num_threads": 8,
        "deterministic_flag": 0,
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    return config


def _infer_port_side(x: float, y: float, canvas_w: float, canvas_h: float) -> str:
    d_left = abs(x - 0.0)
    d_right = abs(canvas_w - x)
    d_bottom = abs(y - 0.0)
    d_top = abs(canvas_h - y)
    dmin = min(d_left, d_right, d_bottom, d_top)
    if dmin == d_top:
        return "TOP"
    if dmin == d_bottom:
        return "BOTTOM"
    if dmin == d_left:
        return "LEFT"
    return "RIGHT"


def _pb_attr_f(key: str, value: float) -> str:
    return (
        "  attr {\n"
        f'    key: "{key}"\n'
        "    value {\n"
        f"      f: {value:.6f}\n"
        "    }\n"
        "  }\n"
    )


def _pb_attr_ph(key: str, value: str) -> str:
    return (
        "  attr {\n"
        f'    key: "{key}"\n'
        "    value {\n"
        f'      placeholder: "{value}"\n'
        "    }\n"
        "  }\n"
    )


def _write_pb_node(
    f: Any,
    name: str,
    node_type: str,
    x: float,
    y: float,
    width: Optional[float] = None,
    height: Optional[float] = None,
    orientation: Optional[str] = None,
    side: Optional[str] = None,
    macro_name: Optional[str] = None,
    x_offset: Optional[float] = None,
    y_offset: Optional[float] = None,
    inputs: Optional[List[str]] = None,
    weight: Optional[float] = None,
) -> None:
    f.write("node {\n")
    f.write(f'  name: "{name}"\n')
    if inputs:
        for s in inputs:
            f.write(f'  input: "{s}"\n')
    f.write(_pb_attr_ph("type", node_type))
    if width is not None:
        f.write(_pb_attr_f("width", width))
    if height is not None:
        f.write(_pb_attr_f("height", height))
    if orientation is not None:
        f.write(_pb_attr_ph("orientation", orientation))
    if side is not None:
        f.write(_pb_attr_ph("side", side))
    f.write(_pb_attr_f("x", x))
    f.write(_pb_attr_f("y", y))
    if macro_name is not None:
        f.write(_pb_attr_ph("macro_name", macro_name))
    if x_offset is not None:
        f.write(_pb_attr_f("x_offset", x_offset))
    if y_offset is not None:
        f.write(_pb_attr_f("y_offset", y_offset))
    if weight is not None:
        f.write(_pb_attr_f("weight", weight))
    f.write("}\n")


def export_plc(
    benchmark_path: str,
    out_dir: str,
    design_name: Optional[str] = None,
    use_pin_nets: bool = True,
) -> Dict[str, Any]:
    data = load_benchmark_any(benchmark_path)
    bench = _normalize_benchmark(data)
    if design_name is None:
        design_name = Path(benchmark_path).stem

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    netlist_path = out_path / "netlist.pb.txt"
    plc_path = out_path / "initial.plc"

    num_macros = bench["num_macros"]
    num_hard = bench["num_hard_macros"]
    num_soft = bench["num_soft_macros"]
    num_ports = int(bench["port_positions"].shape[0])
    canvas_w = float(bench["canvas_width"])
    canvas_h = float(bench["canvas_height"])
    macro_pos = bench["macro_positions"]
    macro_sizes = bench["macro_sizes"]
    macro_fixed = bench["macro_fixed"]
    macro_names = list(bench["macro_names"])
    net_nodes = bench["net_nodes"]
    net_pin_nodes = bench["net_pin_nodes"]
    net_weights = bench["net_weights"]
    macro_pin_offsets = bench["macro_pin_offsets"]

    with netlist_path.open("w", encoding="utf-8") as f:
        # Metadata header expected by PlacementCost parser.
        f.write("node {\n")
        f.write('  name: "__metadata__"\n')
        f.write(
            '  attr {\n    key: "soft_macro_area_bloating_ratio"\n    value {\n      f: 1.0\n    }\n  }\n'
        )
        f.write("}\n")

        plc_indices: Dict[str, int] = {}
        node_idx = 0

        # Ports first (matches many reference testcases; order is not strict).
        for p in range(num_ports):
            name = f"P{p:04d}"
            x = float(bench["port_positions"][p, 0].item())
            y = float(bench["port_positions"][p, 1].item())
            side = _infer_port_side(x, y, canvas_w, canvas_h)
            _write_pb_node(f, name=name, node_type="PORT", x=x, y=y, side=side)
            plc_indices[name] = node_idx
            node_idx += 1

        # Module nodes.
        for i in range(num_macros):
            name = str(macro_names[i])
            x = float(macro_pos[i, 0].item())
            y = float(macro_pos[i, 1].item())
            w = float(macro_sizes[i, 0].item())
            h = float(macro_sizes[i, 1].item())
            if i < num_hard:
                _write_pb_node(
                    f,
                    name=name,
                    node_type="MACRO",
                    x=x,
                    y=y,
                    width=w,
                    height=h,
                    orientation="N",
                )
            else:
                _write_pb_node(
                    f,
                    name=name,
                    node_type="macro",
                    x=x,
                    y=y,
                    width=w,
                    height=h,
                )
            plc_indices[name] = node_idx
            node_idx += 1

        # Pin nodes from hyperedges.
        pin_nodes: List[Dict[str, Any]] = []
        pin_endpoints_per_net: List[List[str]] = []
        use_pin = (
            use_pin_nets
            and len(net_pin_nodes) == len(net_nodes)
            and len(net_pin_nodes) > 0
        )

        for ni in range(len(net_nodes)):
            endpoints: List[Tuple[int, int]] = []
            if use_pin:
                endpoints = [(int(r[0]), int(r[1])) for r in net_pin_nodes[ni].tolist()]
            else:
                endpoints = [(int(owner), 0) for owner in net_nodes[ni].tolist()]
            names: List[str] = []
            for ei, (owner, pin_idx) in enumerate(endpoints):
                if owner >= num_macros:
                    port_idx = owner - num_macros
                    names.append(f"P{port_idx:04d}")
                    continue

                macro_name = str(macro_names[owner])
                pin_name = f"{macro_name}/N{ni:05d}P{ei:02d}"
                mx = float(macro_pos[owner, 0].item())
                my = float(macro_pos[owner, 1].item())
                if owner < num_hard:
                    x_off = 0.0
                    y_off = 0.0
                    if owner < len(macro_pin_offsets):
                        offs = macro_pin_offsets[owner]
                        if len(offs) > 0:
                            slot = pin_idx if 0 <= pin_idx < len(offs) else 0
                            x_off = float(offs[slot, 0].item())
                            y_off = float(offs[slot, 1].item())
                    pin_nodes.append(
                        dict(
                            name=pin_name,
                            node_type="MACRO_PIN",
                            x=mx + x_off,
                            y=my + y_off,
                            macro_name=macro_name,
                            x_offset=x_off,
                            y_offset=y_off,
                            weight=float(net_weights[ni].item())
                            if ni < len(net_weights)
                            else 1.0,
                        )
                    )
                else:
                    pin_nodes.append(
                        dict(
                            name=pin_name,
                            node_type="macro_pin",
                            x=mx,
                            y=my,
                            macro_name=macro_name,
                            weight=float(net_weights[ni].item())
                            if ni < len(net_weights)
                            else 1.0,
                        )
                    )
                names.append(pin_name)
            pin_endpoints_per_net.append(names)

        # Attach one directed star per net (driver -> sinks).
        for ni, names in enumerate(pin_endpoints_per_net):
            if len(names) < 2:
                continue
            driver_idx = 0
            for k, nm in enumerate(names):
                if "/" in nm:
                    driver_idx = k
                    break
            driver_name = names[driver_idx]
            sinks = [nm for j, nm in enumerate(names) if j != driver_idx]
            for pn in pin_nodes:
                if pn["name"] == driver_name:
                    pn["inputs"] = sinks
                    break

        # Emit pin nodes and assign parser indices.
        for pn in pin_nodes:
            _write_pb_node(
                f,
                name=pn["name"],
                node_type=pn["node_type"],
                x=pn["x"],
                y=pn["y"],
                macro_name=pn.get("macro_name"),
                x_offset=pn.get("x_offset"),
                y_offset=pn.get("y_offset"),
                inputs=pn.get("inputs"),
                weight=pn.get("weight"),
            )
            node_idx += 1

    # initial.plc: only PORT/MACRO/macro indices, not *_PIN nodes.
    rows = int(bench["grid_rows"]) if int(bench["grid_rows"]) > 0 else 10
    cols = int(bench["grid_cols"]) if int(bench["grid_cols"]) > 0 else 10
    hard_cnt = num_hard
    soft_cnt = num_soft
    with plc_path.open("w", encoding="utf-8") as f:
        f.write("# Placement file for Circuit Training\n")
        f.write("# Columns : %d  Rows : %d\n" % (cols, rows))
        f.write("# Width : %.6f  Height : %.6f\n" % (canvas_w, canvas_h))
        f.write("# Project : circuit_training\n")
        f.write(f"# Block : {design_name}\n")
        f.write(
            "# Routes per micron, hor : %.3f  ver : %.3f\n"
            % (bench["hroutes_per_micron"], bench["vroutes_per_micron"])
        )
        f.write("# Routes used by macros, hor : 0.0  ver : 0.0\n")
        f.write("# Smoothing factor : 2\n")
        f.write("# Overlap threshold : 0.004\n")
        f.write("# HARD_MACROs     : %9d\n" % hard_cnt)
        f.write("# HARD_MACRO_PINs : %9d\n" % 0)
        f.write("# MACROs          : %9d\n" % (hard_cnt + soft_cnt))
        f.write("# MACRO_PINs      : %9d\n" % 0)
        f.write("# PORTs           : %9d\n" % num_ports)
        f.write("# SOFT_MACROs     : %9d\n" % soft_cnt)
        f.write("# SOFT_MACRO_PINs : %9d\n" % 0)
        f.write("# STDCELLs        : %9d\n" % 0)
        f.write("# node_index x y orientation fixed\n")

        for p in range(num_ports):
            name = f"P{p:04d}"
            idx = plc_indices[name]
            x = float(bench["port_positions"][p, 0].item())
            y = float(bench["port_positions"][p, 1].item())
            f.write(f"{idx} {x:.6f} {y:.6f} - 0\n")

        for i in range(num_macros):
            name = str(macro_names[i])
            idx = plc_indices[name]
            x = float(macro_pos[i, 0].item())
            y = float(macro_pos[i, 1].item())
            orient = "N"
            fixed = 1 if bool(macro_fixed[i].item()) else 0
            f.write(f"{idx} {x:.6f} {y:.6f} {orient} {fixed}\n")

    return {
        "netlist": str(netlist_path),
        "plc": str(plc_path),
        "design_name": design_name,
        "out_dir": str(out_path),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark <-> DREAMPlace Bookshelf bridge"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    to_bs = subparsers.add_parser("to-bookshelf", help="Export Benchmark to Bookshelf")
    to_bs.add_argument("--input", required=True, help="Benchmark .pt file")
    to_bs.add_argument("--out-dir", required=True, help="Output directory")
    to_bs.add_argument(
        "--design-name", default=None, help="Design name for Bookshelf files"
    )
    to_bs.add_argument(
        "--scale", type=float, default=1000.0, help="Micron->integer scale"
    )
    to_bs.add_argument(
        "--row-height-um", type=float, default=1.0, help="Row height in microns"
    )
    to_bs.add_argument(
        "--site-width-um", type=float, default=1.0, help="Site width in microns"
    )
    to_bs.add_argument("--no-ports", action="store_true", help="Exclude IO ports")
    to_bs.add_argument(
        "--no-pin-offsets", action="store_true", help="Ignore pin offsets"
    )
    to_bs.add_argument(
        "--port-size-um", type=float, default=1.0, help="Port size in microns"
    )
    to_bs.add_argument(
        "--write-config", action="store_true", help="Emit DreamPlace config JSON"
    )
    to_bs.add_argument("--bins-x", type=int, default=None, help="Override num_bins_x")
    to_bs.add_argument("--bins-y", type=int, default=None, help="Override num_bins_y")
    to_bs.add_argument(
        "--iterations", type=int, default=200, help="Global placement iterations"
    )
    to_bs.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Global placement learning rate",
    )
    to_bs.add_argument(
        "--target-density", type=float, default=0.8, help="Target density"
    )
    to_bs.add_argument("--gpu", type=int, default=0, help="Use GPU (1) or CPU (0)")
    to_bs.add_argument("--seed", type=int, default=123, help="Random seed")
    to_bs.add_argument(
        "--macro-place-flag", type=int, default=0, help="Enable macro placement mode"
    )

    from_bs = subparsers.add_parser(
        "from-bookshelf", help="Import DREAMPlace .pl into Benchmark"
    )
    from_bs.add_argument("--input", required=True, help="Original Benchmark .pt")
    from_bs.add_argument("--pl", required=True, help="DREAMPlace .pl file")
    from_bs.add_argument("--map", required=True, help="Mapping JSON from to-bookshelf")
    from_bs.add_argument("--output", required=True, help="Output Benchmark .pt")
    from_bs.add_argument(
        "--update-fixed", action="store_true", help="Update fixed flags from .pl"
    )

    to_plc = subparsers.add_parser(
        "to-plc", help="Export Benchmark to PlacementCost files"
    )
    to_plc.add_argument("--input", required=True, help="Benchmark .pt file")
    to_plc.add_argument("--out-dir", required=True, help="Output directory")
    to_plc.add_argument(
        "--design-name", default=None, help="Optional block/design name"
    )
    to_plc.add_argument(
        "--no-pin-nets", action="store_true", help="Use coarse net_nodes only"
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.command == "to-bookshelf":
        result = export_bookshelf(
            benchmark_path=args.input,
            out_dir=args.out_dir,
            design_name=args.design_name,
            scale=args.scale,
            row_height_um=args.row_height_um,
            site_width_um=args.site_width_um,
            include_ports=not args.no_ports,
            include_pin_offsets=not args.no_pin_offsets,
            port_size_um=args.port_size_um,
        )

        if args.write_config:
            bench = _normalize_benchmark(load_benchmark_any(args.input))
            config_path = str(
                Path(args.out_dir) / f"{result['design_name']}.config.json"
            )
            write_dreamplace_config(
                config_path=config_path,
                aux_path=result["aux"],
                canvas_w=bench["canvas_width"],
                canvas_h=bench["canvas_height"],
                grid_rows=bench["grid_rows"],
                grid_cols=bench["grid_cols"],
                num_bins_x=args.bins_x,
                num_bins_y=args.bins_y,
                iterations=args.iterations,
                learning_rate=args.learning_rate,
                target_density=args.target_density,
                gpu=args.gpu,
                seed=args.seed,
                macro_place_flag=args.macro_place_flag,
            )

        print(json.dumps(result, indent=2))

    elif args.command == "from-bookshelf":
        import_bookshelf_solution(
            original_benchmark_path=args.input,
            pl_path=args.pl,
            mapping_path=args.map,
            output_path=args.output,
            update_fixed_from_pl=args.update_fixed,
        )

    elif args.command == "to-plc":
        result = export_plc(
            benchmark_path=args.input,
            out_dir=args.out_dir,
            design_name=args.design_name,
            use_pin_nets=not args.no_pin_nets,
        )
        print(json.dumps(result, indent=2))


# if __name__ == "__main__":
#     main()
