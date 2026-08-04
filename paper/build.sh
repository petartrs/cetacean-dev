#!/usr/bin/env bash
# Build the paper PDF using the bundled tectonic (no system LaTeX needed).
set -e
cd "$(dirname "$0")"
TECTONIC=".tools/tectonic"
if [[ -x "$TECTONIC" ]]; then
  "$TECTONIC" main.tex
elif command -v tectonic >/dev/null 2>&1; then
  tectonic main.tex
else
  echo "tectonic not found (.tools/tectonic missing and not on PATH)" >&2
  exit 1
fi
echo "Built: $(pwd)/main.pdf"
