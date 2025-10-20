#!/usr/bin/env bash
set -euo pipefail

# Config
MANIM_BIN=${MANIM_BIN:-~/miniconda3/envs/manim/bin/manimgl}
VIDEOS_DIR="/Users/d.tarasov/workspace/hse/fintech-dl-hse/course/videos"
export PYTHONPATH="/Users/d.tarasov/workspace/hse/fintech-dl-hse/videos"

usage() {
  echo "Usage: $0 MANIM_FILE.py Scene1 [Scene2 ...] [-o OUTPUT.mp4]" >&2
  echo "  MANIM_FILE.py: Path to the manim file to render" >&2
  echo "  SceneN: One or more scene class names to render and concatenate" >&2
  echo "  -o OUTPUT.mp4: Optional output file path (defaults to VIDEOS_DIR/<MANIM_FILE_BASENAME>.mp4)" >&2
}

if [ "$#" -lt 2 ]; then
  usage
  exit 1
fi

MANIM_FILE="$1"
shift

if [ ! -f "$MANIM_FILE" ]; then
  echo "Error: manim file not found: $MANIM_FILE" >&2
  exit 1
fi

OUTPUT_FILE=""
SCENES=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    -o|--output)
      shift || true
      OUTPUT_FILE="${1:-}"
      if [ -z "$OUTPUT_FILE" ]; then
        echo "Error: -o|--output requires a file path" >&2
        exit 1
      fi
      ;;
    *)
      SCENES+=("$1")
      ;;
  esac
  shift || true
done

if [ "${#SCENES[@]}" -eq 0 ]; then
  echo "Error: Please provide at least one scene name" >&2
  usage
  exit 1
fi

if [ -z "$OUTPUT_FILE" ]; then
  base_name="$(basename "$MANIM_FILE" .py)"
  OUTPUT_FILE="${VIDEOS_DIR}/${base_name}.mp4"
fi

echo "Rendering scenes with ManimGL: ${SCENES[*]}"
"$MANIM_BIN" "$MANIM_FILE" "${SCENES[@]}" --write_file

# Build concat list
LIST_FILE="$(mktemp)"
{
  for scene in "${SCENES[@]}"; do
    scene_file="${VIDEOS_DIR}/${scene}.mp4"
    if [ -f "$scene_file" ]; then
      echo "file '$scene_file'"
    else
      echo "Warning: missing scene output, skipping: $scene_file" >&2
    fi
  done
} > "$LIST_FILE"

echo "Concatenating to: $OUTPUT_FILE"
ffmpeg -hide_banner -loglevel error -y -f concat -safe 0 -i "$LIST_FILE" -c copy "$OUTPUT_FILE"

rm -f "$LIST_FILE"
