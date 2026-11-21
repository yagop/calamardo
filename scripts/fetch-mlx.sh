#!/bin/sh
# Downloads the official prebuilt MLX (libmlx.dylib, mlx.metallib and headers)
# from the mlx-metal wheel on PyPI, verifies its sha256 and unpacks it into
# third_party/mlx. No CMake, no Metal compiler, no Xcode.
#
# The MLX version must match the MLX-C revision vendored in crates/mlx-sys.
set -eu

VERSION=0.32.2
WHEEL=mlx_metal-${VERSION}-py3-none-macosx_26_0_arm64.whl
URL=https://files.pythonhosted.org/packages/dd/cd/4e50bf325100e7165e13d025f264362bf0009196269f9eaf87f2c6e738a2/${WHEEL}
SHA256=e6abeac9ac5265830c9c1541b6f96e9be37a85c2446763a46ad466c63a3837ab

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
