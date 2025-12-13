#!/bin/bash
# Replace image paths to point to local output/ folder

cd "$(dirname "$0")"

sed -E 's|\!\[([^]]*)\]\([^)]*\/([^/)]+)\)|\![\1](output/\2)|g' index.md > index_final.md
echo "Created index_final.md with updated paths"
