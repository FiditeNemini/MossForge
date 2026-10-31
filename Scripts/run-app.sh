#!/usr/bin/env bash
# Build the MossForge SwiftUI app and run it as a minimal .app bundle (bundle id dev.mossforge.app),
# with MOSSFORGE_MODELS pointing at the repo's models/ folder. Dev convenience — the signed/notarised
# app is Scripts/build-app.sh (xcodegen); signing/notarisation is Phase 8.
#   Scripts/run-app.sh            # release build, then launch
#   Scripts/run-app.sh --no-build # relaunch the last build
#   Scripts/run-app.sh --tweak    # release build with -DTWEAKBENCH (a no-op with the bundled stand-in)
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
if [[ "${1:-}" == "--tweak" ]]; then
  (cd MossForgeMLX && swift build -c release --product MossForgeApp -Xswiftc -DTWEAKBENCH)
elif [[ "${1:-}" != "--no-build" ]]; then
  (cd MossForgeMLX && swift build -c release --product MossForgeApp)
fi
APP="$ROOT/MossForgeMLX/.build/MossForge.app"
mkdir -p "$APP/Contents/MacOS"
cp "$ROOT/MossForgeMLX/.build/release/MossForgeApp" "$APP/Contents/MacOS/MossForge"
# SwiftPM resource bundles — above all mlx-swift_Cmlx.bundle (the Metal kernels: without it MLX
# aborts with "Failed to load the default metallib" on the first GPU op). MLX looks next to the
# executable; Bundle.main lookups use Contents/Resources.
mkdir -p "$APP/Contents/Resources"
for b in "$ROOT"/MossForgeMLX/.build/release/*.bundle; do
  rsync -a --delete "$b" "$APP/Contents/MacOS/"
  rsync -a --delete "$b" "$APP/Contents/Resources/"
done
[[ -f "$ROOT/branding/MossForge.icns" ]] || "$ROOT/Scripts/make-icons.sh"
cp "$ROOT/branding/MossForge.icns" "$APP/Contents/Resources/MossForge.icns"
# The dev bundle reports the latest release tag (v0.2.6 → 0.2.6) and the commit count as its build.
VERSION=$(git -C "$ROOT" describe --tags --abbrev=0 2>/dev/null | sed 's/^v//'); VERSION=${VERSION:-0.0.0}
BUILD=$(git -C "$ROOT" rev-list --count HEAD 2>/dev/null || echo 0)
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleIdentifier</key><string>dev.mossforge.app</string>
  <key>CFBundleName</key><string>MossForge</string>
  <key>CFBundleExecutable</key><string>MossForge</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleIconFile</key><string>MossForge</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>CFBundleVersion</key><string>$BUILD</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict></plist>
PLIST
LOG="$ROOT/MossForgeMLX/.build/MossForge.log"
MOSSFORGE_MODELS="${MOSSFORGE_MODELS:-$ROOT/models}" nohup "$APP/Contents/MacOS/MossForge" > "$LOG" 2>&1 &
echo "launched $APP (pid $!, log $LOG)"
