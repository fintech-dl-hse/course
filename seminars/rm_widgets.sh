
for file in ./seminars/*.ipynb; do
    ~/homebrew/bin/jq ' del(.metadata.widgets) ' $file | ~/homebrew/bin/sponge $file
done
