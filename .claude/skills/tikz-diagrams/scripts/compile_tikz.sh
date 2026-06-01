#!/bin/bash
# compile_tikz.sh — Compile a TikZ .tex file to PDF + PNG preview
# Usage: compile_tikz.sh <input.tex> [dpi]
#
# Outputs (sibling to input file):
#   <name>.pdf   — vector PDF
#   <name>.png   — raster preview
#
# Engine + converter are auto-detected so this works on MacTeX/TeX Live as
# well as the Claude.ai sandbox:
#   - LaTeX engine:  $TIKZ_ENGINE (default pdflatex; falls back to xelatex/lualatex)
#   - PNG converter: pdftoppm -> pdftocairo -> Ghostscript (gs) -> sips

set -euo pipefail

INPUT="${1:-}"
DPI="${2:-300}"

if [[ -z "$INPUT" || ! -f "$INPUT" ]]; then
  echo "ERROR: File not found: ${INPUT:-<none>}" >&2
  echo "Usage: compile_tikz.sh <input.tex> [dpi]" >&2
  exit 1
fi

DIR="$(cd "$(dirname "$INPUT")" && pwd)"
BASE="$(basename "$INPUT" .tex)"

cd "$DIR"

# Make sure Homebrew (any common prefix) and MacTeX bins are visible even when
# the skill is launched with a minimal PATH.
for d in "$HOME/homebrew/bin" /opt/homebrew/bin /usr/local/bin /Library/TeX/texbin; do
  [[ -d "$d" ]] && case ":$PATH:" in *":$d:"*) :;; *) PATH="$d:$PATH";; esac
done
export PATH

# --- Pick a LaTeX engine ---------------------------------------------------
ENGINE="${TIKZ_ENGINE:-pdflatex}"
if ! command -v "$ENGINE" >/dev/null 2>&1; then
  for alt in pdflatex xelatex lualatex; do
    if command -v "$alt" >/dev/null 2>&1; then ENGINE="$alt"; break; fi
  done
fi
if ! command -v "$ENGINE" >/dev/null 2>&1; then
  echo "ERROR: no LaTeX engine found on PATH (tried pdflatex/xelatex/lualatex)." >&2
  exit 1
fi

# --- Compile (two passes for any references) -------------------------------
echo "==> Compiling $BASE.tex with $ENGINE ..."
"$ENGINE" -interaction=nonstopmode -halt-on-error "$BASE.tex" > /dev/null 2>&1 || true
if grep -q 'rerun' "$BASE.log" 2>/dev/null; then
  "$ENGINE" -interaction=nonstopmode -halt-on-error "$BASE.tex" > /dev/null 2>&1 || true
fi

if [[ ! -f "$BASE.pdf" ]]; then
  echo "ERROR: PDF compilation failed. Last 30 log lines:" >&2
  tail -30 "$BASE.log" >&2 2>/dev/null || true
  exit 1
fi
echo "==> PDF created: $DIR/$BASE.pdf"

# --- Convert PDF -> PNG with whatever is available -------------------------
png_ok() { [[ -f "$BASE.png" ]]; }

convert_png() {
  if command -v pdftoppm >/dev/null 2>&1; then
    echo "==> Converting to PNG via pdftoppm (${DPI} dpi) ..."
    pdftoppm -png -r "$DPI" -singlefile "$BASE.pdf" "$BASE" && png_ok && return 0
  fi
  if command -v pdftocairo >/dev/null 2>&1; then
    echo "==> Converting to PNG via pdftocairo (${DPI} dpi) ..."
    pdftocairo -png -r "$DPI" -singlefile "$BASE.pdf" "$BASE" && png_ok && return 0
  fi
  if command -v gs >/dev/null 2>&1; then
    echo "==> Converting to PNG via Ghostscript (${DPI} dpi) ..."
    gs -q -dSAFER -dBATCH -dNOPAUSE -dFirstPage=1 -dLastPage=1 \
       -sDEVICE=png16m -r"$DPI" \
       -dTextAlphaBits=4 -dGraphicsAlphaBits=4 \
       -sOutputFile="$BASE.png" "$BASE.pdf" >/dev/null 2>&1 && png_ok && return 0
  fi
  if command -v magick >/dev/null 2>&1; then
    echo "==> Converting to PNG via ImageMagick (magick, ${DPI} dpi) ..."
    magick -density "$DPI" "$BASE.pdf[0]" -background white -flatten "$BASE.png" >/dev/null 2>&1 && png_ok && return 0
  fi
  if command -v convert >/dev/null 2>&1; then
    echo "==> Converting to PNG via ImageMagick (convert, ${DPI} dpi) ..."
    convert -density "$DPI" "$BASE.pdf[0]" -background white -flatten "$BASE.png" >/dev/null 2>&1 && png_ok && return 0
  fi
  if command -v sips >/dev/null 2>&1; then
    echo "==> Converting to PNG via sips (DPI not configurable) ..."
    sips -s format png "$BASE.pdf" --out "$BASE.png" >/dev/null 2>&1 && png_ok && return 0
  fi
  return 1
}

if convert_png; then
  echo "==> PNG created: $DIR/$BASE.png"
else
  echo "WARN: no PDF->PNG converter found on PATH (tried pdftoppm/pdftocairo/gs/magick/convert/sips)." >&2
  echo "WARN: PDF is still available at $DIR/$BASE.pdf (PNG is optional)." >&2
fi

# --- Clean up LaTeX auxiliary files ----------------------------------------
rm -f "$BASE.aux" "$BASE.log" "$BASE.out" "$BASE.nav" "$BASE.snm" \
      "$BASE.toc" "$BASE.fls" "$BASE.fdb_latexmk" "$BASE.synctex.gz"

echo "==> Done."
