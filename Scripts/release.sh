#!/usr/bin/env bash
# Build a signed, notarised, stapled MossForge release: the .app and a .dmg around it.
#
#   Scripts/release.sh <version> [--install] [--skip-notarize]
#     <version>        marketing version, e.g. 1.1.0 (build number = git commit count)
#     --install        also replace /Applications/MossForge.app with the stapled build
#     --skip-notarize  sign + package only (fast local check; Gatekeeper will still warn)
#     --resume-dmg     reuse build.noindex/release/<version>/ (app already notarised) and only redo
#                      the DMG: build, sign, notarise, staple, check
#
# One-time setup per machine (see docs/dev/release.md):
#   - Config/Signing.local.xcconfig with your DEVELOPMENT_TEAM and MOSSFORGE_BUNDLE_ID
#   - the "Developer ID Application" certificate for that team in the login keychain
#   - xcrun notarytool store-credentials <profile> --apple-id <id> --team-id <team>
#   - environment: MOSSFORGE_NOTARY_PROFILE=<profile> (default: mossforge-notary)
#
# Every step stops the script on failure, so a stale or unsigned build is never shipped.
set -euo pipefail
cd "$(dirname "$0")/.."

VERSION="${1:-}"
[[ "$VERSION" =~ ^[0-9]+\.[0-9]+(\.[0-9]+)?$ ]] || {
  echo "usage: Scripts/release.sh <version, e.g. 1.1.0> [--install] [--skip-notarize]" >&2; exit 2; }
shift
INSTALL=0; NOTARIZE=1; RESUME_DMG=0
for arg in "$@"; do
  case "$arg" in
    --install) INSTALL=1 ;;
    --skip-notarize) NOTARIZE=0 ;;
    --resume-dmg) RESUME_DMG=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

TEAM_ID=$(sed -n 's/^DEVELOPMENT_TEAM *= *\([A-Z0-9]*\).*/\1/p' Config/Signing.local.xcconfig 2>/dev/null | tail -1)
[[ -n "$TEAM_ID" ]] || { echo "Set DEVELOPMENT_TEAM in Config/Signing.local.xcconfig (see Config/Signing.local.xcconfig.example)." >&2; exit 1; }
PROFILE=${MOSSFORGE_NOTARY_PROFILE:-mossforge-notary}
IDENTITY="Developer ID Application"
BUILD_NUMBER=$(git rev-list --count HEAD)
# `.noindex` keeps Spotlight from listing every build's MossForge.app beside the installed one.
OUT=build.noindex/release/$VERSION
ARCHIVE=$OUT/MossForge.xcarchive
APP=$OUT/MossForge.app
DMG=$OUT/MossForge-$VERSION.dmg

step() { printf '\n==> %s\n' "$*"; }

notarize() {  # notarize <file> — submit, wait, and print Apple's log if it's rejected
  local file=$1 result id
  result=$(xcrun notarytool submit "$file" --keychain-profile "$PROFILE" --wait --output-format json)
  echo "$result"
  id=$(/usr/bin/plutil -extract id raw - <<<"$result")
  if [[ "$(/usr/bin/plutil -extract status raw - <<<"$result")" != "Accepted" ]]; then
    xcrun notarytool log "$id" --keychain-profile "$PROFILE" >&2 || true
    echo "notarization of $(basename "$file") failed" >&2
    exit 1
  fi
}

# Notarisation can take most of an hour; keep the Mac (and display) awake so the screen
# doesn't lock — a locked login keychain hides the notarytool profile mid-run.
caffeinate -di -w $$ &

step "Preflight"
# The certificate's SHA-1, so signing never picks an ambiguous match.
SIGN_HASH=$(security find-identity -v -p codesigning | grep "$IDENTITY: .*($TEAM_ID)" | head -1 | awk '{print $2}')
[[ -n "$SIGN_HASH" ]] || { echo "No '$IDENTITY' certificate for team $TEAM_ID in the keychain." >&2; exit 1; }
if (( NOTARIZE )); then
  xcrun notarytool history --keychain-profile "$PROFILE" >/dev/null || {
    echo "No notarytool profile '$PROFILE' — see the setup notes at the top of this script." >&2; exit 1; }
fi
[[ -z "$(git status --porcelain)" ]] || echo "warning: working tree is dirty — the build won't match a commit"
echo "MossForge $VERSION ($BUILD_NUMBER) from $(git rev-parse --short HEAD)"
if (( RESUME_DMG )); then
  xcrun stapler validate "$APP" || { echo "no stapled app in $OUT to resume from" >&2; exit 1; }
else
rm -rf "$OUT"; mkdir -p "$OUT"

step "xcodegen generate"
xcodegen generate

step "Archive (Release, Developer ID)"
xcodebuild -project MossForge.xcodeproj -scheme MossForgeApp-macOS -configuration Release \
  -destination 'generic/platform=macOS' -archivePath "$ARCHIVE" \
  -derivedDataPath build.noindex/xcode-release \
  ARCHS=arm64 \
  MARKETING_VERSION="$VERSION" CURRENT_PROJECT_VERSION="$BUILD_NUMBER" \
  archive > "$OUT/archive.log" 2>&1 || { tail -40 "$OUT/archive.log" >&2; exit 1; }

step "Export"
sed "s/__TEAM_ID__/$TEAM_ID/" Scripts/release/ExportOptions.plist > "$OUT/ExportOptions.plist"
xcodebuild -exportArchive -archivePath "$ARCHIVE" -exportPath "$OUT" \
  -exportOptionsPlist "$OUT/ExportOptions.plist" > "$OUT/export.log" 2>&1 \
  || { tail -40 "$OUT/export.log" >&2; exit 1; }

step "Verify signature"
codesign --verify --deep --strict --verbose=2 "$APP"
details=$(codesign -dvv "$APP" 2>&1)
grep -q "Authority=$IDENTITY: .*($TEAM_ID)" <<<"$details" || { echo "not signed with $IDENTITY" >&2; exit 1; }
grep -q "flags=.*runtime" <<<"$details" || { echo "hardened runtime missing" >&2; exit 1; }
grep -q "^Timestamp=" <<<"$details" || { echo "secure timestamp missing" >&2; exit 1; }
CLI=$APP/Contents/MacOS/moss-mlx
[[ -x "$CLI" ]] || { echo "moss-mlx missing from the app bundle" >&2; exit 1; }
cli=$(codesign -dvv "$CLI" 2>&1)
grep -q "Authority=$IDENTITY: .*($TEAM_ID)" <<<"$cli" || { echo "moss-mlx not signed with $IDENTITY" >&2; exit 1; }
grep -q "flags=.*runtime" <<<"$cli" || { echo "moss-mlx hardened runtime missing" >&2; exit 1; }
"$CLI" smoke | grep -q "^30.0$" || { echo "moss-mlx smoke failed inside the bundle" >&2; exit 1; }
/usr/libexec/PlistBuddy -c "Print CFBundleShortVersionString" "$APP/Contents/Info.plist"

if (( NOTARIZE )); then
  step "Notarise + staple the app"
  ditto -c -k --keepParent "$APP" "$OUT/MossForge.zip"
  notarize "$OUT/MossForge.zip"
  xcrun stapler staple "$APP"
  rm "$OUT/MossForge.zip"
fi
fi  # RESUME_DMG

step "Build the DMG"
STAGE=$OUT/dmg
rm -rf "$STAGE" "$DMG"; mkdir -p "$STAGE"
ditto "$APP" "$STAGE/MossForge.app"
ln -s /Applications "$STAGE/Applications"
hdiutil create -volname "MossForge $VERSION" -srcfolder "$STAGE" -fs HFS+ -format UDZO -ov "$DMG" >/dev/null
rm -rf "$STAGE"
codesign --sign "$SIGN_HASH" --timestamp "$DMG"

if (( NOTARIZE )); then
  step "Notarise + staple the DMG"
  notarize "$DMG"
  xcrun stapler staple "$DMG"

  step "Gatekeeper check"
  spctl --assess --type execute --verbose=2 "$APP"
  spctl --assess --type open --context context:primary-signature --verbose=2 "$DMG"
  xcrun stapler validate "$APP"
  xcrun stapler validate "$DMG"
fi

if (( INSTALL )); then
  step "Install to /Applications"
  pgrep -x MossForge >/dev/null && { echo "MossForge is running — quit it first." >&2; exit 1; }
  rm -rf /Applications/MossForge.app
  ditto "$APP" /Applications/MossForge.app
fi

step "Done"
echo "app: $APP"
echo "dmg: $DMG"
echo "Tag it when happy:  git tag v$VERSION && git push --tags (once there's a remote)"
