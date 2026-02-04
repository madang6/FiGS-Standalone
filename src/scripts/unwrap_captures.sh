#!/usr/bin/env bash

# src/scripts/unwrap_captures.sh

CAPTURE_DIR="/home/admin/StanfordMSL/FiGS-Standalone/3dgs/capture"

for d in "$CAPTURE_DIR"/*/; do
    id="$(basename "$d")"

    mp4="$(ls "$d"/*.[mM][pP]4 2>/dev/null | head -n 1)"
    if [ -n "$mp4" ]; then
        cp "$mp4" "$CAPTURE_DIR/$id.mp4"
    fi
done
