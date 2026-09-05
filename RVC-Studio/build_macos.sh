#!/bin/zsh
set -euo pipefail

cd "${0:A:h}"
export CLANG_MODULE_CACHE_PATH="${TMPDIR:-/private/tmp}/rvc-studio-clang-cache"
export SWIFTPM_MODULECACHE_OVERRIDE="${TMPDIR:-/private/tmp}/rvc-studio-swift-cache"

swift build -c release --product RVCStudio --scratch-path .build

APP="${PWD}/build/RVC-Studio.app"
BINARY="${PWD}/.build/release/RVCStudio"
rm -rf "${APP}"
mkdir -p "${APP}/Contents/MacOS" "${APP}/Contents/Resources"
cp "${BINARY}" "${APP}/Contents/MacOS/RVCStudio"
cp "${PWD}/Resources/Info.plist" "${APP}/Contents/Info.plist"
cp "${PWD}/Resources/RVCStudio.icns" "${APP}/Contents/Resources/RVCStudio.icns"

codesign --force --deep --sign - "${APP}"
codesign --verify --deep --strict "${APP}"
print "Built native SwiftUI app: ${APP}"
