#!/bin/bash
# Downloads the PJRT CPU plugin that JAX-CIP is tested with and checks its hash.
# jaxlib 0.10 links its CPU client statically and ships no plugin library, so the plugin comes
# from the prebuilt artifacts of zml/pjrt-artifacts, release v18.0.0 (PJRT C API 0.114).
#
# Usage: scripts/fetch_pjrt_plugin.sh DIRECTORY
# Afterwards: export JAXCIP_PJRT_PLUGIN=DIRECTORY/libpjrt_cpu.so
set -euo pipefail

DIRECTORY=$(realpath -m "$1")
URL=https://github.com/zml/pjrt-artifacts/releases/download/v18.0.0/pjrt-cpu_linux-amd64.tar.gz
SHA256=edc5ecc1b13eceeba4b0c312a05efbfdd9bd0732040990024beacb64cc9c4569

mkdir -p "$DIRECTORY"
curl -fsSL "$URL" | tar -xz -C "$DIRECTORY"
PLUGIN=$(find "$DIRECTORY" -name libpjrt_cpu.so | head -n 1)
if [ "$PLUGIN" != "$DIRECTORY/libpjrt_cpu.so" ]; then
  mv "$PLUGIN" "$DIRECTORY/libpjrt_cpu.so"
fi
echo "$SHA256  $DIRECTORY/libpjrt_cpu.so" | sha256sum --check
