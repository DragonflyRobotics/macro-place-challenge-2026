import torch

from macro_place.benchmark import Benchmark
from dream.dreamplace_bridge import save_benchmark_any
from dream.dreamplace_pipeline import main
import subprocess


class SimpleRandomPlacer:
    def __init__(self):
        pass

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        save_benchmark_any(benchmark, "run.pt")
        # print("Placing macros randomly...")
        # result = subprocess.run(["ls"], capture_output=True, text=True)
        # print(result.stdout)
        return main()
