#!/bin/sh
# Downloads the official prebuilt MLX (libmlx.dylib, mlx.metallib and headers)
# from the mlx-metal wheel on PyPI, verifies its sha256 and unpacks it into
# third_party/mlx. No CMake, no Metal compiler, no Xcode.
#
# The MLX version must match the MLX-C revision vendored in crates/mlx-sys.
set -eu

VERSION=0.32.4
WHEEL=mlx_metal-${VERSION}-py3-none-macosx_26_0_arm64.whl
URL=https://files.pythonhosted.org/packages/2a/d6/4340ed6e184f71693d7f531c9ae55d677451351245760d71f8dd714dcbe5/${WHEEL}
SHA256=514df92d41a286895d2564b4b18f4da45096bd7a4a0c47e1042f345ad5a7f936

ROOT=$(cd "$(dirname "$0")/.." && pwd)
DEST=${CALAMARDO_MLX_DIR:-$ROOT/third_party/mlx}

if [ -f "$DEST/VERSION" ] && [ "$(cat "$DEST/VERSION")" = "$VERSION" ]; then
    echo "MLX $VERSION already in $DEST"
    exit 0
fi

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "Downloading $WHEEL"
curl -fL --retry 3 -o "$TMP/$WHEEL" "$URL"
echo "$SHA256  $TMP/$WHEEL" | shasum -a 256 -c -

mkdir -p "$TMP/x"
unzip -q "$TMP/$WHEEL" 'mlx/lib/*' 'mlx/include/*' -d "$TMP/x"

rm -rf "$DEST"
mkdir -p "$DEST"
mv "$TMP/x/mlx/lib" "$TMP/x/mlx/include" "$DEST/"
echo "$VERSION" > "$DEST/VERSION"
echo "MLX $VERSION installed in $DEST"
