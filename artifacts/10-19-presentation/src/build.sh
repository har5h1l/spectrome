#!/bin/zsh
set -eu
base="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$base/build/pdf-pages"
cd "$base/src"
/Library/TeX/texbin/latexmk -norc -pdf -interaction=nonstopmode -halt-on-error -synctex=1 -outdir="$base/build" main.tex > "$base/build/compile-console.txt" 2>&1
cp "$base/build/main.pdf" "$base/presentation.pdf"
pdftoppm -scale-to 1600 -png "$base/presentation.pdf" "$base/build/pdf-pages/page"
