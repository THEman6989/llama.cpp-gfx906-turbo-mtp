# llama.cpp GFX906 Turbo Patch Wrapper Makefile

TARGET_DIR=llama.cpp-gfx906-turbo
PATCH_FILE=turbo-gfx906-mtp.patch
LOG_FILE=patch-apply.log

.PHONY: all patch build clean help

all: patch build

patch:
	@echo "Starting patching process..."
	@AUTO_YES=1 bash apply-turbo.sh

build:
	@echo "Starting build process for GFX906..."
	@if [ ! -d "$(TARGET_DIR)" ]; then \
		echo "Error: Target directory $(TARGET_DIR) not found. Run 'make patch' first."; \
		exit 1; \
	fi
	@mkdir -p $(TARGET_DIR)/build
	@cd $(TARGET_DIR)/build && \
		cmake .. -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 -DGGML_HIP_GRAPHS=ON && \
		make -j llama-cli

clean:
	@echo "Cleaning up..."
	@rm -rf $(TARGET_DIR) $(LOG_FILE)
	@echo "Done."

help:
	@echo "Available commands:"
	@echo "  make patch   - Clones upstream and applies the GFX906 Turbo patch"
	@echo "  make build   - Compiles the patched repository for GFX906"
	@echo "  make all     - Runs both patch and build"
	@echo "  make clean   - Deletes the patched repository and logs"
