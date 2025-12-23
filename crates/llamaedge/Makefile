# LlamaEdge Rust SDK - Build Makefile
#
# This Makefile provides convenient commands for building the SDK
# within the monorepo, where explicit target specification is required.

# Detect host platform
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_S),Darwin)
    ifeq ($(UNAME_M),arm64)
        HOST_TARGET := aarch64-apple-darwin
    else
        HOST_TARGET := x86_64-apple-darwin
    endif
else ifeq ($(UNAME_S),Linux)
    ifeq ($(UNAME_M),aarch64)
        HOST_TARGET := aarch64-unknown-linux-gnu
    else
        HOST_TARGET := x86_64-unknown-linux-gnu
    endif
else
    HOST_TARGET := x86_64-pc-windows-msvc
endif

.PHONY: build build-release test check clean doc

build:
	cargo build --target $(HOST_TARGET)

build-release:
	cargo build --release --target $(HOST_TARGET)

test:
	cargo test --target $(HOST_TARGET)

check:
	cargo check --target $(HOST_TARGET)

clippy:
	cargo +nightly clippy --target $(HOST_TARGET) --all-features -- -D warnings

doc:
	cargo doc --target $(HOST_TARGET) --no-deps --open

clean:
	cargo clean