#!/bin/bash
# setup.sh — run once after cloning/unzipping to verify structure
# No action needed; files are already in place.
# This script just confirms the expected structure exists.

echo "Forget Me Not — Story Repository"
echo "================================="
echo ""

dirs=(
  "timeline"
  "chapters/act-01"
  "chapters/act-02"
  "characters/amy"
  "characters/leo"
  "characters/nick"
  "characters/abhi"
  "characters/tanu"
  "characters/miles"
  "characters/vic"
  "characters/ash"
  "pov"
  "flashbacks"
  "places"
  "notes"
)

all_good=true

for dir in "${dirs[@]}"; do
  if [ -d "$dir" ]; then
    echo "✓ $dir"
  else
    echo "✗ MISSING: $dir"
    all_good=false
  fi
done

echo ""
if [ "$all_good" = true ]; then
  echo "All directories present. Ready to write."
else
  echo "Some directories missing. Check above."
fi
