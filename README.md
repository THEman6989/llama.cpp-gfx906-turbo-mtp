# llama.cpp

![llama](https://user-images.githubusercontent.com/1991296/230134379-7181e485-c521-4d23-a0d6-f7b3b61ba524.png)

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp)](https://github.com/ggml-org/llama.cpp/releases)
[![Server](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml/badge.svg)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)

[Manifesto](https://github.com/ggml-org/llama.cpp/discussions/205) / [ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md)

LLM inference in C/C++

## Recent API changes

- [Changelog for `libllama` API](https://github.com/ggml-org/llama.cpp/issues/9289)
- [Changelog for `llama-server` REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

## Hot topics

- **Hugging Face cache migration: models downloaded with `-hf` are now stored in the standard Hugging Face cache directory, enabling sharing with other HF tools.**
- **[guide : using the new WebUI of llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/16938)**
- [guide : running gpt-oss with llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/15396)
- [[FEEDBACK] Better packaging for llama.cpp to support downstream consumers 🤗](https://github.com/ggml-org/llama.cpp/discussions/15313)
- Support for the `gpt-oss` model with native MXFP4 format has been added | [PR](https://github.com/ggml-org/llama.cpp/pull/15091) | [Collaboration with NVIDIA](https://blogs.nvidia.com/blog/rtx-ai-garage-openai-oss) | [Comment](https://github.com/ggml-org/llama.cpp/discussions/15095)
- Multimodal support arrived in `llama-server`: [#12898](https://github.com/ggml-org/llama.cpp/pull/12898) | [documentation](./docs/multimodal.md)
- VS Code extension for FIM completions: https://github.com/ggml-org/llama.vscode
- Vim/Neovim plugin for FIM completions: https://github.com/ggml-org/llama.vim
- Hugging Face Inference Endpoints now support GGUF out of the box! https://github.com/ggml-org/llama.cpp/discussions/9669
- Hugging Face GGUF editor: [discussion](https://github.com/ggml-org/llama.cpp/discussions/9268) | [tool](https://huggingface.co/spaces/CISCai/gguf-editor)

----

## Quick start

Getting started with llama.cpp is straightforward. Here are several ways to install it on your machine:

- Install `llama.cpp` using [brew, nix or winget](docs/install.md)
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed, you'll need a model to work with. Head to the [Obtaining and quantizing models](#obtaining-and-quantizing-models) section to learn more.

Example command:

```sh
# Use a local model file
llama-cli -m my_model.gguf

# Or download and run a model directly from Hugging Face
llama-cli -hf ggml-org/gemma-3-1b-it-GGUF

# Launch OpenAI-compatible API server
llama-server -hf ggml-org/gemma-3-1b-it-GGUF
```

## Description

The main goal of `llama.cpp` is to enable LLM inference with minimal setup and state-of-the-art performance on a wide
range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is the main playground for developing new features for the [ggml](https://github.com/ggml-org/ggml) library.

<details>
<summary>Models</summary>

Typically finetunes of the base models below are supported as well.

Instructions for adding support for new models: [HOWTO-add-model.md](docs/development/HOWTO-add-model.md)

#### Text-only

- [X] LLaMA 🦙
- [x] LLaMA 2 🦙🦙
- [x] LLaMA 3 🦙🦙🦙
- [X] [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [x] [Mixtral MoE](https://huggingface.co/models?search=mistral-ai/Mixtral)
- [x] [DBRX](https://huggingface.co/databricks/dbrx-instruct)
- [x] [Jamba](https://huggingface.co/ai21labs)
- [X] [Falcon](https://huggingface.co/models?search=tiiuae/falcon)
- [X] [Chinese LLaMA / Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca) and [Chinese LLaMA-2 / Alpaca-2](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2)
- [X] [Vigogne (French)](https://github.com/bofenghuang/vigogne)
- [X] [BERT](https://github.com/ggml-org/llama.cpp/pull/5423)
- [X] [Koala](https://bair.berkeley.edu/blog/2023/04/03/koala/)
- [X] [Baichuan 1 & 2](https://huggingface.co/models?search=baichuan-inc/Baichuan) + [derivations](https://huggingface.co/hiyouga/baichuan-7b-sft)
- [X] [Aquila 1 & 2](https://huggingface.co/models?search=BAAI/Aquila)
- [X] [Starcoder models](https://github.com/ggml-org/llama.cpp/pull/3187)
- [X] [Refact](https://huggingface.co/smallcloudai/Refact-1_6B-fim)
- [X] [MPT](https://github.com/ggml-org/llama.cpp/pull/3417)
- [X] [Bloom](https://github.com/ggml-org/llama.cpp/pull/3553)
- [x] [Yi models](https://huggingface.co/models?search=01-ai/Yi)

# llama.cpp-gfx906-turbo-mtp

# GFX906 Turbo + MTP Support (Experimental)

This repository is a merge of the latest [llama.cpp](https://github.com/ggml-org/llama.cpp) (with MTP/Eagle support) and the [GFX906 Turbo fork](https://github.com/arte-fact/llamacpp-gfx-906-turbo).

**WARNING: This is an experimental port.** While the kernels and logic have been verified to match the Turbo fork, full stability with the latest upstream changes is not yet guaranteed.

## Features
- **Latest Upstream Support:** Includes Multi-Token Prediction (MTP), Eagle, Medusa, and the new speculative decoding framework.
- **GFX906 Wave64 Kernels:** High-performance kernels for AMD MI50/MI60 (Radeon VII).
- **TurboQuant:** Support for 2-bit, 3-bit (turbo3), and 4-bit KV cache compression.
- **Shadow Cache:** Workaround for ROCm 6.0+ stability issues on GFX906.

## Build Instructions

```bash
mkdir build && cd build
cmake .. -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 -DGGML_HIP_GRAPHS=ON
make -j llama-cli
```

---


# llama.cpp-gfx906-turbo

llama.cpp fork for **AMD MI50/MI60 (gfx906)** with **TurboQuant turbo3 KV cache compression**.

Combines upstream llama.cpp with [iacopPBK](https://github.com/iacopPBK/llama.cpp-gfx906) gfx906 Wave64 kernels and [Madreag](https://github.com/Madreag/turbo3-cuda) TurboQuant CUDA port, plus 9 HIP-specific bug fixes for gfx906 correctness.

## Results

- **3.3x more context** than fp16 KV cache on the same hardware
- **4.6x KV cache compression** (turbo3: 3.5 bits/value vs 16 bits fp16)
- **300K context** with turbo3 vs **90K max** with f16 on Qwen3-Coder-Next 80B
- **1M context** on Qwen3.5-27B Q4_0 with 4x MI50
- **Matches vllm-gfx906** performance at 56-57 tok/s on MoE models

### Benchmark: Qwen3-Coder-Next Q4_1 (80B MoE) — 4x MI50 (64 GB)

| KV Cache | Max Context | Gen (tok/s) | Prompt (tok/s) | Peak VRAM |
|----------|-------------|-------------|----------------|-----------|
| **turbo3 K+V** | **300,000** | 37.2 | 116.1 | 97% |
| turbo3 K+V | 256,000 | 39.7 | 130.8 | 93% |
| f16 K+V + HIP graphs | 90,000 | 57.4 | 153.2 | 96% |
| f16 K+V | 65,536 | 51.6 | 153.2 | 96% |

Tensor split: `7,8,8,8` — flags: `--no-mmap --no-warmup`

**Trade-off**: turbo3 gives 3.3x more context at 18% gen speed cost (at same context size).

### Speed by Model Type

| Model | Config | Gen tok/s | Why |
|-------|--------|-----------|-----|
| MoE 80B (3B active) | f16 + HIP graphs | 57 | Small weight reads, no pipeline impact |
| Dense 9B (1 GPU) | turbo3 K+V shadow | 59 | No pipeline bubbles, shadow cache |
| Dense 24B (4 GPU PP) | turbo3 K+V shadow | 21.5 | Pipeline parallel overhead |
| Dense 27B (4 GPU PP) | turbo3 K+V shadow | 18.5 | Large model + pipeline bubbles |

**Key insight**: gfx906 at batch=1 is **latency-bound** (10% bandwidth utilization), not bandwidth-bound. Both this fork and vllm-gfx906 hit the same ~56 tok/s ceiling on MoE models. Dense models suffer from pipeline parallelism bubbles. Tensor parallelism (`--split-mode row`) crashes on HIP because it requires P2P GPU access — vllm solves this via RCCL AllReduce over shared memory, which llama.cpp doesn't implement.

### gfx906-Optimized Quant Types

Only these quantization formats get the optimized warp-cooperative MMVQ kernel:

| Quant | gfx906 MMVQ | Notes |
|-------|-------------|-------|
| **Q4_0** | YES | Fastest for decode (small + optimized) |
| **Q4_1** | YES | Slightly larger, same speed |
| **Q8_0** | YES | Best quality, simpler dequant |
| Q4_K_M | no | Falls to generic path — slower |
| Q5_K, Q6_K | no | Generic path |
| IQ types | no | Generic path |

**Use Q4_0 or Q4_1 for best speed.** Q4_K_M is common but unoptimized on gfx906.

## Performance Tuning

Enable HIP graphs for +8-10% gen speed:
```bash
cmake .. -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 -DCMAKE_BUILD_TYPE=Release -DGGML_HIP_GRAPHS=ON
```

**Speculative decoding** for pure transformer models (+47% warm cache):
```bash
--spec-type ngram-mod --spec-ngram-size-n 24 --draft-min 8 --draft-max 24
```
Note: doesn't work on hybrid SSM+attention models (Qwen3-Coder-Next, Qwen3.5).

### gfx906 Kernel Tuning (from vllm-gfx906 analysis)
- `num_stages=1` everywhere (LDS stability)
- `num_warps=4` (register pressure sweet spot)
- LDS limit 64KB (not 160KB like newer chips)
- Adaptive block sizes for small batches

## Build

```bash
cd llama-cpp-gfx906-turbo
mkdir build && cd build
cmake .. -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 -DCMAKE_BUILD_TYPE=Release -DGGML_HIP_GRAPHS=ON
cmake --build . --target llama-server -j$(nproc)
```

Requires ROCm 7.1+.

## Usage

```bash
# Recommended: turbo3 K+V for maximum context
./build/bin/llama-server \
  -m model.gguf \
  --host 0.0.0.0 --port 8080 \
  -ngl 999 -c 262144 -np 1 \
  --cache-type-k turbo3 --cache-type-v turbo3 \
  --no-warmup

# Alternative: turbo3 K + f16 V (faster on dense models, less compression)
  --cache-type-k turbo3 --cache-type-v f16
```

For context beyond 262K, add YaRN:
```bash
  --rope-scaling yarn --rope-freq-scale 0.25 \
  --yarn-ext-factor 1.0 --yarn-attn-factor 1.0 \
  --yarn-beta-fast 32 --yarn-beta-slow 1
```

## Changes from Upstream

### New Files

**gfx906 Wave64 kernels** (from iacopPBK) — `ggml/src/ggml-cuda/gfx906/`
- 29 files: DPP warp reductions, Q8 FlashAttention tile kernel, optimized RoPE, warp-cooperative MMVQ, software-pipelined MMQ, custom SGEMM/MMF for medium batch sizes

**TurboQuant CUDA kernels** (from Madreag)
- `ggml/src/ggml-cuda/turbo-quant.cu` + `.cuh` — FWHT rotation, SET_ROWS quantize, dequant kernels for turbo2/3/4
- `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-turbo*` — 5 FA template instances
- `ggml/src/ggml-turbo-quant.c` — CPU quant/dequant
- `src/turbo-rotation-data.h` — Precomputed 128x128 rotation matrix

### Modified Files — gfx906 Patches

| File | Change |
|------|--------|
| `ggml-cuda/common.cuh` | DPP shuffle XOR dispatch, `GGML_CUDA_CC_IS_GCN` |
| `ggml-cuda/fattn-common.cuh` | GCN tile sizes, turbo3 KQ v_dot2 + V dequant with LUT cache |
| `ggml-cuda/fattn.cu` | Q8 tile kernel + turbo3 shadow cache + native vec path |
| `ggml-cuda/fattn-vec.cuh` | Turbo3 FA vec dispatch, sparse V skip, `__expf` |
| `ggml-cuda/mmq.cu` + `mmq.cuh` | Vectorized loads, software pipelining for GCN |
| `ggml-cuda/mmvq.cu` | Half-warp MoE dispatch, Wave64 warp ID fix |
| `ggml-cuda/quantize.cu` | DPP-based warp reductions for Q8 quantize |
| `ggml-cuda/rope.cu` | gfx906 RoPE kernel dispatch |
| `ggml-cuda/vecdotq.cuh` | gfx906 MXFP4 lookup, `get_int_b2_fast` |
| `ggml-cuda/add-id.cu` | gfx906 vectorized add_id |
| `ggml-cuda/mmid.cu` | gfx906 MoE dispatch |
| `ggml-cuda/ssm-scan.cu` | gfx906 SSM optimization |
| `ggml-cuda/ggml-cuda.cu` | gfx906 custom GEMM dispatch, SOLVE_TRI limits, turbo3 ops |
| `ggml-hip/CMakeLists.txt` | gfx906 sources, turbo3 FA instances, `-fPIC`, `-O1` for turbo-quant |

### Modified Files — TurboQuant Type System

| File | Change |
|------|--------|
| `ggml/include/ggml.h` | `GGML_TYPE_TURBO2/3/4_0` enum, `GGML_OP_TURBO_WHT` |
| `ggml/src/ggml.c` | Type table entries, `ggml_turbo_wht()` op constructor |
| `ggml/src/ggml-common.h` | `block_turbo3_0` struct (16 bytes per 32 values) |
| `ggml/src/ggml-quants.h` | `quantize_row_turbo3_0_ref`, `dequantize_row_turbo3_0` |
| `ggml-cuda/convert.cu` | Turbo3 dequant dispatch (fp16/fp32/bf16) |
| `ggml-cuda/set-rows.cu` | Turbo3 quantize dispatch (KV cache write) |
| `ggml-cuda/getrows.cu` | Turbo3 get_rows dispatch |
| `ggml-cuda/cpy.cu` + `cpy-utils.cuh` | Turbo3 copy + quantize block function |
| `ggml-cuda/dequantize.cuh` | `dequantize_turbo3_0` for get_rows |

### Modified Files — llama.cpp Integration

| File | Change |
|------|--------|
| `src/llama-graph.cpp` | WHT rotation for Q (forward) and V output (inverse) in `build_attn` |
| `src/llama-kv-cache.cpp` + `.h` | Rotation matrix tensor init, layer-adaptive mode |
| `src/llama-context.cpp` | Force flash attention for turbo types |
| `src/llama-memory.h` + hybrid variants | Turbo rotation accessors |
| `common/arg.cpp` | `--cache-type-k turbo3` CLI flag |
| `tools/llama-bench/llama-bench.cpp` | Turbo type name strings |

### Modified Files — CPU + Metal Backends

| File | Change |
|------|--------|
| `ggml-cpu/ggml-cpu.c` | Turbo type traits, `GGML_OP_TURBO_WHT` dispatch |
| `ggml-cpu/ops.cpp` + `ops.h` | CPU FWHT with K and V sign arrays |
| `ggml-cpu/quants.h` | CPU quant function declarations |
| `ggml-metal/ggml-metal-device.cpp/.h/.m` | Turbo WHT Metal pipeline |
| `ggml-metal/ggml-metal-ops.cpp/.h` | Turbo WHT Metal dispatch |
| `ggml-metal/ggml-metal-impl.h` | Turbo WHT kernel args struct |

### HIP/gfx906 Bug Fixes

9 bugs found and fixed during integration:

| Bug | Fix | File |
|-----|-----|------|
| `TURBO_WHT` falls through to `SOLVE_TRI` dimension check | Add explicit `return true` before `SOLVE_TRI` | `ggml-cuda.cu` |
| Shadow cache V dequant reads wrong strides for 4D view | Use native FA vec kernel (handles all layouts) | `fattn.cu` |
| `hipMalloc` in WHT dispatch deadlocks on multi-GPU | Use FWHT kernel (no device alloc) | `turbo-quant.cu` |
| HIP compiler misoptimizes FWHT butterfly at `-O3` | Compile `turbo-quant.cu` at `-O1` | `CMakeLists.txt` |
| `cuda_fp16.h` not found on HIP | Add `#ifdef GGML_USE_HIP` guard | `turbo-quant.cu` |
| Missing WHT rotation in `build_attn(attn_kv_iswa)` | Add turbo WHT calls matching `attn_kv` overload | `llama-graph.cpp` |
| HIP objects not position-independent | Add `-fPIC` to HIP compile flags | `CMakeLists.txt` |
| CPU WHT uses K signs for V inverse rotation | Add V-specific sign arrays (`turbo_wht_s1_v/s2_v`) | `ops.cpp` |
| Shadow cache non-contiguous layout breaks FA | Use contiguous strides (ne1 not capacity) + full re-dequant | `fattn.cu` |

## Known Limitations

- **Tensor parallelism**: `--split-mode row` crashes on HIP/ROCm — requires P2P GPU access which MI50 on PCIe doesn't support. vllm solves this with RCCL AllReduce over shared memory. Dense models on multi-GPU use pipeline parallelism (bubble overhead at batch=1).
- **Hybrid SSM models**: May hang during warmup on gfx906 due to `SOLVE_TRI` — use `--no-warmup`.
- **Speculative decoding**: Only works on pure transformer models (Devstral), not hybrid SSM+attention (Qwen3.5, Qwen3-Coder-Next).
- **YaRN extended context**: Server slot context cap removed — full allocated context usable with YaRN rope scaling.

## Next Steps

1. **TP4 on ROCm** — requires RCCL integration for cross-GPU AllReduce → 2x dense model speed
2. **Speculative decoding for hybrid SSM** — needs SSM state checkpointing for rollback
3. **Fused MoE kernels** — `gfx906/fused/` has RMS+mul+MMQ fusion, reduce kernel launch count
4. **Incremental shadow cache** — current full re-dequant every pass; incremental would amortize cost

## Credits

- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) — Base
- [iacopPBK/llama.cpp-gfx906](https://github.com/iacopPBK/llama.cpp-gfx906) — gfx906 Wave64 kernels
- [Madreag/turbo3-cuda](https://github.com/Madreag/turbo3-cuda) — TurboQuant CUDA port
- [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) — TurboQuant CPU + Metal
- [Aaryan-Kapoor/llama.cpp](https://github.com/Aaryan-Kapoor/llama.cpp/tree/turboquant-tq3_0) — Clean CPU TurboQuant
- [ai-infos/vllm-gfx906-mobydick](https://github.com/ai-infos/vllm-gfx906-mobydick) — vllm gfx906 reference
- TurboQuant paper: [arXiv 2504.19874](https://arxiv.org/abs/2504.19874) (ICLR 2026)
