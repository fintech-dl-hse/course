usage() {

  echo "Usage: $0 VIDEO_FILE AUDIO_FILE OUTPUT_FILE" >&2
  echo "  VIDEO_FILE: Path to the video file" >&2
  echo "  AUDIO_FILE: Path to the audio file" >&2
  echo "  OUTPUT_FILE: Path to the output file" >&2
}

if [ "$#" -ne 3 ]; then
  usage
  exit 1
fi

VIDEO_FILE="$1"
AUDIO_FILE="$2"
OUTPUT_FILE="$3"

ffmpeg -i "$VIDEO_FILE" -i "$AUDIO_FILE" -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 "$OUTPUT_FILE"