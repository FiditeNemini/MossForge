# Releasing MossForge (signed + notarised)

`Scripts/release.sh <version>` produces a Developer ID–signed, notarised, stapled
`MossForge.app` and a `MossForge-<version>.dmg` around it, under `build.noindex/release/<version>/` (the `.noindex` suffix keeps Spotlight from
listing build copies of the app; local dev builds go to `build.noindex/xcode/`).
Both pass Gatekeeper with no warning, online or offline.

```sh
Scripts/release.sh 1.1.0                  # full release
Scripts/release.sh 1.1.0 --install        # … and replace /Applications/MossForge.app
Scripts/release.sh 1.1.0 --skip-notarize  # sign + package only (quick signing check)
Scripts/release.sh 1.1.0 --resume-dmg     # app already notarised: redo only the DMG steps
```

## One-time setup per machine
1. Your signing identity: copy `Config/Signing.local.xcconfig.example` to
   `Config/Signing.local.xcconfig` (gitignored) and set `DEVELOPMENT_TEAM` (your Apple team ID) and
   `MOSSFORGE_BUNDLE_ID` (e.g. `com.yourname.mossforge`). Without it, Debug builds are ad-hoc signed
   with the placeholder bundle id `com.example.mossforge`.
2. The **Developer ID Application** certificate for that team in the login keychain
   (Xcode → Settings → Accounts → Manage Certificates). `security find-identity -v -p
   codesigning` must list it.
3. A notarytool keychain profile. Create an app-specific password at account.apple.com
   (Sign-In and Security), then:
   ```sh
   xcrun notarytool store-credentials mossforge-notary --apple-id <apple-id> --team-id <team-id>
   ```
   Paste the password at the prompt (never on the command line). The profile lives in that
   Mac's login keychain only — repeat on each release machine. The script uses the profile named
   in `MOSSFORGE_NOTARY_PROFILE` (default `mossforge-notary`); check it with
   `xcrun notarytool history --keychain-profile mossforge-notary`.

## What the script does
1. **Preflight** — finds the certificate (by SHA-1, so the match is never ambiguous) and the
   notary profile; warns on a dirty tree. Version = the argument; build number = commit count.
2. **Archive** — `xcodegen generate`, then `xcodebuild archive` (Release, `ARCHS=arm64` on the
   command line so the Swift packages build arm64-only too — MLX is Apple-silicon only and the
   engine's `Float16` code doesn't compile for x86_64).
3. **Export** — `developer-id` method (`Scripts/release/ExportOptions.plist`, with your team filled in).
4. **Verify** — strict deep `codesign --verify`, and asserts the Developer ID authority, the
   hardened runtime flag, and a secure timestamp — the three things notarisation rejects.
5. **Notarise + staple the app**, then build the DMG (app + `/Applications` link), sign it,
   **notarise + staple the DMG**. A rejection prints Apple's log and stops.
6. **Gatekeeper check** — `spctl --assess` on both, `stapler validate` on both.

Signing config lives in `project.yml` and `Config/Signing.xcconfig`: hardened runtime on, **Debug
ad-hoc** (no certificate needed day to day), **Release** manual "Developer ID Application" with
`--timestamp`. No entitlements file: the app is not sandboxed (it reads a models folder
anywhere and runs a local server) and needs no hardened-runtime exceptions.

## Notes
- Notarisation can take anything from a few minutes to most of an hour . The script runs `caffeinate -di` for its lifetime: a locked screen
  locks the login keychain, notarytool then can't read its profile ("No Keychain password item
  found"), and the run stops. If that happens after the app is stapled, use `--resume-dmg`.
- A locally built DMG isn't quarantined, so double-clicking it here never shows Gatekeeper —
  `spctl` is the real test. To see the user's path, download the DMG from somewhere (AirDrop to
  the other Mac works) and open it.
- `moss-mlx` ships inside the app at `MossForge.app/Contents/MacOS/moss-mlx` (XcodeGen `moss-mlx`
  tool target, embedded + signed by the app target), so the app's notarisation covers it. It
  finds MLX's metallib in the app's Resources. Launched through a symlink it re-execs itself by
  its real path first — MLX locates the bundle from the launch path. The Verify step checks it
  is present, Developer ID–signed with the hardened runtime, and passes `moss-mlx smoke`.
- `hdiutil create` prints a deprecation warning on macOS 27; it still works.
