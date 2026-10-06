#!/bin/zsh
# Rebuild the deck from its existing evidence. Does not run an experiment.
set -eu
base="$(cd "$(dirname "$0")/.." && pwd)"
runtime_root='/Users/harshilshah/.cache/codex-runtimes/codex-primary-runtime/dependencies'
export RUNTIME_NODE_MODULES="$runtime_root/node/node_modules"
export RUNTIME_NODE="$runtime_root/node/bin/node"
export RUNTIME_PYTHON="$runtime_root/python/bin/python3"
export RUNTIME_BIN_DIR="$runtime_root/bin/override"
mkdir -p "$base/build/pdf-pages"
"$runtime_root/python/bin/python3" -c 'from pathlib import Path; import sys; [p.unlink() for p in Path(sys.argv[1]).glob("page-*.png")]' "$base/build/pdf-pages"
cd "$base/src"
/Library/TeX/texbin/latexmk -norc -pdf -interaction=nonstopmode -halt-on-error -synctex=1 -outdir="$base/build" main.tex > "$base/build/compile-console.txt" 2>&1
"$runtime_root/bin/override/pdftoppm" -scale-to 1600 -png "$base/build/main.pdf" "$base/build/pdf-pages/page"
"$runtime_root/python/bin/python3" "$base/src/prepare_exports.py"
ln -sfn "$runtime_root/node/node_modules" "$base/build/node_modules"
cp "$base/src/export_pptx.mjs" "$base/build/export_pptx.mjs"
"$runtime_root/node/bin/node" "$base/build/export_pptx.mjs" > "$base/build/export-console.json"
