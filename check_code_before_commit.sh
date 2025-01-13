#!/bin/bash

# Format code
cargo +nightly fmt --all -- --check

# Clippy
cargo +nightly clippy --target wasm32-wasip1 --all-features -- -D warnings
