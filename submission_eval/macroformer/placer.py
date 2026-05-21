import torch

from macro_place.benchmark import Benchmark
from dream.dreamplace_bridge import save_benchmark_any


class SimpleRandomPlacer:
    def __init__(self):
        pass

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        return benchmark.macro_positions
