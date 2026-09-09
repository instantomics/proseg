#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACKAGE_ROOT="$ROOT/packaging/proseg_bin"
BUILD_ROOT="${TMPDIR:?TMPDIR must identify allocation-local storage}/proseg-bin-build"
INSTALL_ROOT="$BUILD_ROOT/install"

rm -rf "$BUILD_ROOT" "$PACKAGE_ROOT/src/proseg_bin/bin"
mkdir -p "$INSTALL_ROOT" "$PACKAGE_ROOT/src/proseg_bin/bin"

if [[ "$(rustc --version)" != "rustc 1.88.0 "* ]]; then
  echo "Rust 1.88.0 is required" >&2
  exit 1
fi

CARGO_HOME="$BUILD_ROOT/cargo-home" \
CARGO_TARGET_DIR="$BUILD_ROOT/cargo-target" \
cargo install --locked --version 3.2.0 --root "$INSTALL_ROOT" proseg
cp "$INSTALL_ROOT/bin/proseg" "$PACKAGE_ROOT/src/proseg_bin/bin/proseg"
strip "$PACKAGE_ROOT/src/proseg_bin/bin/proseg"

"${PYTHON:-python}" -m build --wheel --outdir "$ROOT/dist" "$PACKAGE_ROOT"
"$PACKAGE_ROOT/src/proseg_bin/bin/proseg" --version
sha256sum "$ROOT"/dist/proseg_bin-3.2.0.post1-py3-none-manylinux_2_34_x86_64.whl
