#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Initializing submodules..."
git -C "$SCRIPT_DIR" submodule update --init --recursive

echo "==> Installing xfac-quad extern..."
cd "$SCRIPT_DIR/extern/xfac_quad"
bash install_extern.sh

echo "Now run: bash compile.sh"