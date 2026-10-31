#!/usr/bin/env bash
# Render the MossForge icon and cut the macOS icon set from it:
#   branding/icon-1024.png                      (Scripts/render-icon.swift)
#   branding/Assets.xcassets/AppIcon.appiconset (Xcode / xcodegen build)
#   branding/MossForge.icns                     (Scripts/run-app.sh dev bundle)
set -euo pipefail
cd "$(dirname "$0")/.."
swift Scripts/render-icon.swift branding/icon-1024.png
SET=branding/Assets.xcassets/AppIcon.appiconset
ICONSET=$(mktemp -d)/MossForge.iconset
mkdir -p "$SET" "$ICONSET"
images=""
for spec in 16:1 16:2 32:1 32:2 128:1 128:2 256:1 256:2 512:1 512:2; do
  pt=${spec%%:*}; scale=${spec##*:}; px=$((pt * scale))
  suffix=""; [[ $scale == 2 ]] && suffix="@2x"
  name="icon_${pt}x${pt}${suffix}.png"
  sips -z "$px" "$px" branding/icon-1024.png --out "$SET/$name" >/dev/null
  cp "$SET/$name" "$ICONSET/$name"
  images+="    {\"idiom\": \"mac\", \"size\": \"${pt}x${pt}\", \"scale\": \"${scale}x\", \"filename\": \"$name\"},"$'\n'
done
printf '{\n  "images": [\n%s  ],\n  "info": {"version": 1, "author": "xcode"}\n}\n' "${images%,$'\n'}"$'\n' > "$SET/Contents.json"
printf '{\n  "info": {"version": 1, "author": "xcode"}\n}\n' > branding/Assets.xcassets/Contents.json
iconutil -c icns "$ICONSET" -o branding/MossForge.icns
echo "icons: $SET, branding/MossForge.icns"
