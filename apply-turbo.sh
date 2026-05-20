#!/usr/bin/env bash

# GFX906 Turbo + MTP Support (Wrapper Script)
# This script applies the Turbo optimizations and GFX906 Wave64 kernels
# to a known-stable version of the upstream llama.cpp repository.

set -e

REPO_URL="https://github.com/ggml-org/llama.cpp.git"
STABLE_COMMIT="acd604fb277044e07c2bff01f4c169167b45f478"
TARGET_DIR="llama.cpp-gfx906-turbo"
PATCH_FILE="turbo-gfx906-mtp.patch"

# Colors for logging
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

function log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

function log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

function log_err() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 1. Ensure patch file exists
if [ ! -f "$PATCH_FILE" ]; then
    log_err "Patch file '$PATCH_FILE' not found in the current directory."
    exit 1
fi

# 2. Clone or update repository
if [ -d "$TARGET_DIR" ]; then
    log_info "Directory '$TARGET_DIR' already exists."
    cd "$TARGET_DIR"
    
    # Check if there are uncommitted changes or active patches
    if ! git diff --quiet || ! git diff --cached --quiet; then
        log_warn "Uncommitted changes found in '$TARGET_DIR'."
        read -p "Do you want to reset the directory to the stable commit? This will DESTROY local changes! [y/N]: " reset_choice
        if [[ "$reset_choice" =~ ^[Yy]$ ]]; then
            git reset --hard HEAD
            git clean -fd
        else
            log_err "Cannot proceed with uncommitted changes."
            exit 1
        fi
    fi
    
    # Fetch latest if needed, though we only need the specific commit
    git fetch origin
else
    log_info "Cloning upstream llama.cpp repository..."
    git clone "$REPO_URL" "$TARGET_DIR"
    cd "$TARGET_DIR"
fi

# 3. Checkout the stable commit
log_info "Checking out stable commit: $STABLE_COMMIT"
git checkout "$STABLE_COMMIT"

# 4. Apply the patch
log_info "Applying GFX906 Turbo patch..."
# Use --reject to save rejected hunks to .rej files if it fails, but fail the script
if git apply --reject "../$PATCH_FILE"; then
    log_info "Patch applied successfully!"
else
    log_err "Failed to apply patch."
    log_err "The upstream repository may have changed significantly."
    log_err "Check the generated '*.rej' files to see which parts of the patch failed."
    log_err "You can manually resolve these conflicts and generate a new patch."
    exit 1
fi

log_info "Repository is ready in './$TARGET_DIR'."
log_info "You can now build the project with:"
echo ""
echo "    cd $TARGET_DIR"
echo "    mkdir build && cd build"
echo "    cmake .. -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 -DGGML_HIP_GRAPHS=ON"
echo "    make -j llama-cli"
echo ""
