#!/usr/bin/env bash
# Build the Mac App Store package: archive the sandboxed MossForgeApp-AppStore target and export it for
# App Store Connect (or upload it directly with --upload).
#
#   Scripts/release-appstore.sh <version> [--upload]
#     <version>  marketing version, e.g. 1.0.0 (build number = git commit count; it must rise every upload)
#     --upload   export with destination "upload" — sends the build to App Store Connect
#
# One-time setup (see docs/dev/release.md, "Mac App Store"):
#   - Config/Signing.local.xcconfig with your DEVELOPMENT_TEAM and MOSSFORGE_BUNDLE_ID
#   - the app record for that bundle id in App Store Connect
#   - Xcode signed in to the team's Apple account (Xcode → Settings → Accounts): automatic signing creates the
#     Apple Distribution certificate and the Mac App Store provisioning profile (-allowProvisioningUpdates)
#
# No notarisation: App Store builds are checked by App Review instead.
set -euo pipefail
cd "$(dirname "$0")/.."

VERSION="${1:-}"
[[ "$VERSION" =~ ^[0-9]+\.[0-9]+(\.[0-9]+)?$ ]] || { echo "usage: Scripts/release-appstore.sh <version> [--upload]" >&2; exit 2; }
shift
DESTINATION=export
for arg in "$@"; do
  case "$arg" in
    --upload) DESTINATION=upload ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

TEAM_ID=$(sed -n 's/^DEVELOPMENT_TEAM *= *\([A-Z0-9]*\).*/\1/p' Config/Signing.local.xcconfig 2>/dev/null | tail -1)
[[ -n "$TEAM_ID" ]] || { echo "Set DEVELOPMENT_TEAM in Config/Signing.local.xcconfig (see Config/Signing.local.xcconfig.example)." >&2; exit 1; }
BUILD_NUMBER=$(git rev-list --count HEAD)
OUT=build.noindex/appstore/$VERSION
ARCHIVE=$OUT/MossForge.xcarchive
mkdir -p "$OUT"
step() { printf '\n==> %s\n' "$*"; }

[[ -z "$(git status --porcelain)" ]] || echo "warning: the working tree has uncommitted changes" >&2

step "xcodegen generate"
xcodegen generate

step "Archive (MossForgeApp-AppStore, Release, sandboxed)"
xcodebuild -project MossForge.xcodeproj -scheme MossForgeApp-AppStore -configuration Release \
  -destination 'generic/platform=macOS' -archivePath "$ARCHIVE" \
  -derivedDataPath build.noindex/xcode-appstore-release -allowProvisioningUpdates \
  ARCHS=arm64 MARKETING_VERSION="$VERSION" CURRENT_PROJECT_VERSION="$BUILD_NUMBER" \
  archive > "$OUT/archive.log" 2>&1 || { tail -40 "$OUT/archive.log" >&2; exit 1; }

step "Check the archive: sandboxed, privacy manifest, no embedded CLI"
APP="$ARCHIVE/Products/Applications/MossForge.app"
ents=$(codesign -d --entitlements - --xml "$APP" 2>/dev/null)
grep -q "com.apple.security.app-sandbox" <<<"$ents" || { echo "archive is not sandboxed" >&2; exit 1; }
[[ -f "$APP/Contents/Resources/PrivacyInfo.xcprivacy" ]] || { echo "PrivacyInfo.xcprivacy missing" >&2; exit 1; }
[[ ! -e "$APP/Contents/MacOS/moss-mlx" ]] || { echo "moss-mlx must not ship in the App Store build" >&2; exit 1; }

step "Export ($DESTINATION, app-store-connect)"
cat > "$OUT/ExportOptions.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>method</key><string>app-store-connect</string>
    <key>teamID</key><string>$TEAM_ID</string>
    <key>signingStyle</key><string>automatic</string>
    <key>destination</key><string>$DESTINATION</string>
</dict>
</plist>
PLIST
xcodebuild -exportArchive -archivePath "$ARCHIVE" -exportPath "$OUT" -allowProvisioningUpdates \
  -exportOptionsPlist "$OUT/ExportOptions.plist" > "$OUT/export.log" 2>&1 \
  || { tail -40 "$OUT/export.log" >&2; exit 1; }

step "Done"
if [[ "$DESTINATION" == upload ]]; then
  echo "uploaded $VERSION ($BUILD_NUMBER) to App Store Connect — it appears under TestFlight after processing"
else
  echo "package: $(ls "$OUT"/*.pkg 2>/dev/null || echo "$OUT")"
  echo "upload it with Transporter, or rerun with --upload"
fi
