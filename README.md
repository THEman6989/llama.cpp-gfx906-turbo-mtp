# llama.cpp GFX906 Turbo + MTP Support (Patch Wrapper)

This repository provides a reliable, patch-based wrapper to inject **GFX906 (Radeon VII / MI50) Turbo optimizations** and **TurboQuant KV Cache Compression** into a known-stable version of the upstream [llama.cpp](https://github.com/ggml-org/llama.cpp) repository.

By using a patch approach, you get the latest features (like MTP, Medusa, and Eagle speculative decoding) without having to manually maintain a heavily modified fork.

## What is included in this patch?
1. **GFX906 Wave64 Kernels:** Highly optimized warp-cooperative kernels tailored specifically for the Radeon VII / MI50 hardware architecture, drastically improving Prompt Processing and Token Generation speed.
2. **TurboQuant:** Support for 2-bit, 3-bit (`turbo3`), and 4-bit KV cache compression to save up to 78% of Context VRAM with minimal quality loss.
3. **Shadow Cache:** A persistent FP16 shadow cache workaround to resolve the known ROCm 6.0+ instability issues on GFX906 during FlashAttention dequantization.
4. **FWHT Rotation:** A fast $O(d \log d)$ Walsh-Hadamard Transform kernel (`GGML_OP_TURBO_WHT`) to rotate the KV cache into a compression-friendly space.
5. **HIP Graphs:** Fully integrated and activated `-DGGML_HIP_GRAPHS=ON` to reduce CPU overhead during decoding.

## How to use

We provide a **Makefile** and a simple Bash script to automate everything. The process is fully logged to `patch-apply.log`.

### 1. Run Everything (Patch + Build)

```bash
make all
```

### 2. Manual Steps

If you want more control, you can run the steps individually:

*   **Apply the patch only:**
    ```bash
    make patch
    ```
*   **Build only (after patching):**
    ```bash
    make build
    ```

### Logging

All output from Git (cloning, checking out, patching) is redirected to **`patch-apply.log`**. If something goes wrong, check this file first.

## Maintaining and Updating

This patch is tied to a specific upstream commit (`acd604fb277044e07c2bff01f4c169167b45f478`). 

If you want to update to a newer upstream commit in the future:
1. Change the `STABLE_COMMIT` variable in `apply-turbo.sh`.
2. Run `make patch`.
3. If the script fails (because upstream code changed significantly), Git will generate `*.rej` files indicating which parts of the patch failed.
4. Manually fix the `.rej` conflicts in the `llama.cpp-gfx906-turbo` directory.
5. Create a new patch using `git diff > turbo-gfx906-mtp.patch` and overwrite the old one.