#!/bin/bash
# Extract image paths from index.md and copy to output/

cd "$(dirname "$0")"
mkdir -p output

# Extract paths from markdown image syntax ![...](path)
grep -oP '\!\[.*?\]\(\K[^)]+' index.md | while read -r path; do
    filename=$(basename "$path")
    cp "$path" "output/$filename"
    echo "Copied: $path -> output/$filename"
done
