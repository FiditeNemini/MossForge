#!/usr/bin/env bash
# Build + test (gate) and format-lint (advisory) for the MossForge root package.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> swift build"
swift build

echo "==> swift test"
swift test

echo "==> swift format lint (advisory)"
swift format lint --recursive Sources Tests \
  || echo "(format lint reported issues — advisory only, not failing the gate)"

echo "OK — MossForge is green."
