#!/usr/bin/env bash
# Execution commands for CacheSlide example.
# Run from the repo root: bash run_cacheslide.sh [hotpotqa|msc|swebench]
set -euo pipefail

cd "$(dirname "$0")"

# 1) Syntax-check the two patched files (fast smoke test).
python - <<'PY'
import ast
for f in [
    "vllm/model_executor/models/llama.py",
    "vllm/worker/model_runner.py",
]:
    ast.parse(open(f).read())
    print(f"parse OK: {f}")

# Quick sanity check: CoPEPositionalEncoder behaves on its own.
import importlib.util, pathlib, sys, torch
spec = importlib.util.spec_from_file_location(
    "_llama_check", "vllm/model_executor/models/llama.py"
)
# We can't import the module fully (depends on vllm runtime), so just
# eval the class block in isolation by extracting it.
src = pathlib.Path("vllm/model_executor/models/llama.py").read_text()
start = src.index("class CoPEPositionalEncoder")
end = src.index("\nclass ", start + 1)
ns = {"torch": torch, "nn": torch.nn, "Tuple": tuple}
exec("import torch.nn as nn\n" + src[start:end], ns)
CoPE_PE = ns["CoPEPositionalEncoder"]
enc = CoPE_PE(fixed_ranges=[(0, 8), (8, 20)])
assert enc.get_range(0) == (0, 8) and enc.get_range(1) == (8, 20)
assert enc.assign(2, 5) == (20, 25)
out = enc.apply(torch.zeros(4, dtype=torch.long), chunk_id=1)
assert out.tolist() == [8, 9, 10, 11], out.tolist()
print("CoPEPositionalEncoder smoke test OK")
PY

# 2) Pick dataset preset (default: hotpotqa).
DATASET="${1:-hotpotqa}"

# 3) Run the harness.
#    --gpu_mem_util may need lowering on smaller GPUs.
python examples/CacheSlide.py \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --gpu_mem_util 0.5 \
    --dataset "${DATASET}" \
    --split validation \
    --num_samples 5 \
    --max_chunks 4 \
    --max_new_tokens 16 \
    --start_index 0 \
    --seed 0

# Per-dataset variants:
# bash run_cacheslide.sh hotpotqa
# bash run_cacheslide.sh msc
# bash run_cacheslide.sh swebench
