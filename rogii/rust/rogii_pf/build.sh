#!/usr/bin/env bash
# Build & install rogii_pf as a CPython extension module via maturin.
# Run from the crate dir: ./build.sh
# After this completes you can `python -c "import rogii_pf"` from anywhere.

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR"

# M1 Pro flags. opt-level/lto already set in Cargo.toml [profile.release];
# this just nails target-cpu in case Cargo's per-target rustflags isn't honored
# under the outer driver.
export RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=apple-m1"

# `--release` => /Users/.../target/release. PyO3 abi3 means we don't need a per-Python build.
maturin develop --release "$@"
