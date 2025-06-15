
for file in ./seminars/*.ipynb; do
    jq ' del(.metadata.widgets) ' $file | sponge $file
done
