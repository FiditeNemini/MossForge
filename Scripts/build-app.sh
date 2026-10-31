#!/usr/bin/env bash
# Generate the Xcode project from project.yml and build MossForge.app (Debug, ad-hoc signed).
#   Scripts/build-app.sh            # build → build.noindex/xcode/Build/Products/Debug/MossForge.app
#   Scripts/build-app.sh --open     # …and launch it
set -euo pipefail
cd "$(dirname "$0")/.."
command -v xcodegen >/dev/null 2>&1 || { echo "xcodegen not found: brew install xcodegen" >&2; exit 1; }
[[ -d branding/Assets.xcassets/AppIcon.appiconset ]] || Scripts/make-icons.sh
echo "==> xcodegen generate"
xcodegen generate --quiet
echo "==> xcodebuild (MossForgeApp-macOS, Debug)"
xcodebuild -project MossForge.xcodeproj -scheme MossForgeApp-macOS -configuration Debug \
  -destination 'platform=macOS,arch=arm64' -derivedDataPath build.noindex/xcode -quiet build
# `.noindex`: Spotlight skips the folder, so dev builds don't show up next to an installed app.
APP=build.noindex/xcode/Build/Products/Debug/MossForge.app
echo "OK — $APP"
[[ "${1:-}" == "--open" ]] && open "$APP"
exit 0
