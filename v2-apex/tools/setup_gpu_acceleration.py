"""Apex Hyperion GPU & Numba LLVM Auto-Setup & Speedup Benchmark (`tools/setup_gpu_acceleration.py`).

Detects NVIDIA CUDA GPUs vs AMD/Intel DirectML GPUs vs Multi-Core CPU (Numba LLVM AVX2/AVX-512),
optionally installs matching GPU wheels into `.venv`, calibrates empirical crossover, and prints
a live speedup benchmark table.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def detect_nvidia_cuda_major() -> int | None:
    """Query nvidia-smi for driver CUDA UMD version (e.g. 13 or 12)."""
    try:
        res = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
        if res.returncode != 0:
            return None
        m = re.search(r"CUDA\s+(?:UMD\s+)?Version:\s*(\d+)\.", res.stdout)
        return int(m.group(1)) if m else 12
    except Exception:
        return None


def detect_directml_adapter() -> str | None:
    """Check Win32_VideoController for non-NVIDIA discrete/integrated GPU."""
    try:
        res = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_VideoController).Name"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in res.stdout.splitlines():
            name = line.strip()
            if name and "NVIDIA" not in name.upper() and "BASIC" not in name.upper():
                return name
    except Exception:
        pass
    return None


def install_packages(no_install: bool = False) -> None:
    if no_install:
        return
    # Ensure numba is always installed
    subprocess.run([sys.executable, "-m", "pip", "install", "numba>=0.60.0", "--quiet"], check=False)
    cuda_major = detect_nvidia_cuda_major()
    if cuda_major is not None:
        pkg = "cupy-cuda13x[ctk]" if cuda_major >= 13 else "cupy-cuda12x[ctk]"
        print(f"[*] Detected NVIDIA GPU (CUDA {cuda_major}.x) -> installing {pkg}...")
        subprocess.run([sys.executable, "-m", "pip", "install", pkg, "--quiet"], check=False)
    else:
        dml = detect_directml_adapter()
        if dml:
            print(f"[*] Detected DirectML-capable GPU ({dml}) -> installing torch-directml...")
            subprocess.run([sys.executable, "-m", "pip", "install", "torch-directml", "--quiet"], check=False)
        else:
            print("[*] CPU-only environment detected -> using Multi-Core Numba LLVM AVX2/AVX-512.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Apex Hyperion GPU/Numba Auto-Detector & Benchmark")
    parser.add_argument("--no-install", action="store_true", help="Skip pip install and run benchmark only")
    args = parser.parse_args()

    install_packages(no_install=args.no_install)

    from apex.compute.hyperion_fabric import ComputeBackend, get_hyperion_fabric

    fabric = get_hyperion_fabric()
    prof = fabric.calibrate_crossover(persist=True)

    console = Console()
    table = Table(title="APEX HYPERION — HARDWARE & SPEEDUP BENCHMARK TABLE", show_lines=True)
    table.add_column("Workload", style="cyan", no_wrap=True)
    table.add_column("CPU Numba LLVM (ms)", justify="right", style="green")
    table.add_column("GPU CUDA Tensor (ms)", justify="right", style="magenta")
    table.add_column("Optimal Route", style="bold yellow")

    table.add_row(
        "1 Symbol x 10,000 Paths",
        f"{prof.bench_1x10k_cpu_ms:.3f}",
        f"{prof.bench_1x10k_gpu_ms:.3f}" if fabric.has_cuda else "N/A",
        ComputeBackend.CUDA_TENSOR.value if (fabric.has_cuda and prof.bench_1x10k_gpu_ms < prof.bench_1x10k_cpu_ms) else ComputeBackend.NUMBA_LLVM_AVX.value,
    )
    table.add_row(
        "12 Symbols x 10,000 Paths",
        f"{prof.bench_12x10k_cpu_ms:.3f}",
        f"{prof.bench_12x10k_gpu_ms:.3f}" if fabric.has_cuda else "N/A",
        ComputeBackend.CUDA_TENSOR.value if (fabric.has_cuda and prof.bench_12x10k_gpu_ms < prof.bench_12x10k_cpu_ms) else ComputeBackend.NUMBA_LLVM_AVX.value,
    )
    table.add_row(
        "12 Symbols x 100,000 Paths",
        f"{prof.bench_12x100k_cpu_ms:.3f}",
        f"{prof.bench_12x100k_gpu_ms:.3f}" if fabric.has_cuda else "N/A",
        ComputeBackend.CUDA_TENSOR.value if (fabric.has_cuda and prof.bench_12x100k_gpu_ms < prof.bench_12x100k_cpu_ms) else ComputeBackend.NUMBA_LLVM_AVX.value,
    )
    console.print(table)
    console.print(
        f"[bold]GPU:[/bold] {prof.gpu_name} | [bold]Empirical N_crossover:[/bold] {prof.n_crossover_paths:,} paths | "
        f"[bold]Saved:[/bold] trader/semantic_memory.json"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
