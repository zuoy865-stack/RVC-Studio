#!/bin/zsh
set -euo pipefail

# RVC_PORTABLE_PYTHON must point to an extracted arm64
# python-build-standalone tree containing bin/python3.
if [[ -z "${RVC_PORTABLE_PYTHON:-}" || ! -x "${RVC_PORTABLE_PYTHON}/bin/python3" ]]; then
  print -u2 "Set RVC_PORTABLE_PYTHON to an arm64 python-build-standalone directory."
  exit 2
fi

cd "${0:A:h}"
./build_macos.sh

APP="${PWD}/build/RVC-Studio.app"
RESOURCES="${APP}/Contents/Resources"
CORE_SOURCE="${PWD}/../RVC-Core"
CORE_TARGET="${RESOURCES}/RVC-Core"

rm -rf "${CORE_TARGET}"
mkdir -p "${CORE_TARGET}/runtime/site-packages"
rsync -a --exclude 'runtime' --exclude '__pycache__' --exclude 'tests' --exclude 'logs/*' \
  "${CORE_SOURCE}/" "${CORE_TARGET}/"
ditto "${RVC_PORTABLE_PYTHON}" "${CORE_TARGET}/runtime/python"
"${CORE_TARGET}/runtime/python/bin/python3" -m pip install --no-compile --target "${CORE_TARGET}/runtime/site-packages" \
  -r "${CORE_TARGET}/requirements-macos.txt"
find "${CORE_TARGET}/runtime/site-packages" "${CORE_TARGET}" -type d -name '__pycache__' -prune -exec rm -rf {} +
codesign --force --deep --sign - "${CORE_TARGET}/runtime/python/bin/python3"
codesign --force --deep --sign - "${APP}"
print "Self-contained app: ${APP}"
