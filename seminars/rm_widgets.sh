
for file in ./seminars/*/*.ipynb; do
    [ -f "$file" ] || continue
    jq ' del(.metadata.widgets) ' "$file" | sponge "$file"
done
