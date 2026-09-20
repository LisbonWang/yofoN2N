"""Peak GPU memory during the TensorRT engine build (separate from runtime).

The builder allocates tactic scratch on the device; that peak can exceed the
engine's runtime footprint.  Sampling torch.cuda.mem_get_info() in a thread is
enough because TRT allocates through the driver, not through torch's caching
allocator.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import bench_trt_engine as B  # noqa: E402

import torch  # noqa: E402


def main() -> None:
    torch.cuda.init()
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    base_used = (total - free) / 1024**2

    peak = {"used": base_used}
    stop = threading.Event()

    def sample() -> None:
        while not stop.is_set():
            f, t = torch.cuda.mem_get_info()
            peak["used"] = max(peak["used"], (t - f) / 1024**2)
            time.sleep(0.05)

    th = threading.Thread(target=sample, daemon=True)
    th.start()
    t0 = time.perf_counter()
    info = B._build_engine(B.ONNX_FP16, B.ENGINE_FP16, B.WORKSPACE_BYTES, fp16=True)
    build_s = time.perf_counter() - t0
    stop.set()
    th.join(timeout=1)

    rep = {
        "base_used_MiB": base_used,
        "peak_used_MiB": peak["used"],
        "peak_delta_MiB": peak["used"] - base_used,
        "post_build_delta_MiB": (torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]) / 1024**2 - base_used,
        "build_s": build_s,
        "workspace_cap_MiB": B.WORKSPACE_BYTES / 1024**2,
        "engine_MiB": info["engine_MiB"],
    }
    with open(B.REPORT, encoding="utf-8") as fh:
        prev = json.load(fh)
    prev["build_peak_memory"] = rep
    with open(B.REPORT, "w", encoding="utf-8") as fh:
        json.dump(prev, fh, indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
