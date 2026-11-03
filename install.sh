#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GLAFIC_VERSION="${GLAFIC_VERSION:-v2.1.15}"
GLAFIC_COMMIT="${GLAFIC_COMMIT:-0b5d303cf8bc0680a377853e1fe03e00f5f3127d}"
BUILD_ROOT="$ROOT/.build"
VENV="$ROOT/.venv"

log() { printf '\n==> %s\n' "$*"; }
fail() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

if [[ "$(uname -s)" == "Darwin" ]]; then
    command -v brew >/dev/null 2>&1 || fail "Homebrew is required on macOS. Install it from https://brew.sh/ and rerun this script."
    log "Installing macOS build dependencies"
    brew update
    brew install python@3.12 gcc make pkg-config cfitsio fftw gsl curl
    PYTHON="$(brew --prefix python@3.12)/bin/python3.12"
    LIBPATH="$(brew --prefix)/lib"
    INCPATH="$(brew --prefix)/include"
elif [[ "$(uname -s)" == "Linux" ]]; then
    if command -v apt-get >/dev/null 2>&1; then
        SUDO=()
        [[ "${EUID}" -eq 0 ]] || SUDO=(sudo)
        log "Installing Linux build dependencies"
        "${SUDO[@]}" apt-get update
        "${SUDO[@]}" apt-get install -y build-essential ca-certificates curl \
            pkg-config python3 python3-venv python3-pip \
            libcfitsio-dev libfftw3-dev libgsl-dev
        PYTHON="$(command -v python3)"
    elif command -v dnf >/dev/null 2>&1; then
        SUDO=()
        [[ "${EUID}" -eq 0 ]] || SUDO=(sudo)
        log "Installing Fedora/RHEL build dependencies"
        "${SUDO[@]}" dnf install -y gcc gcc-c++ make ca-certificates curl \
            pkgconf-pkg-config python3 python3-pip \
            cfitsio-devel fftw-devel gsl-devel
        PYTHON="$(command -v python3)"
    elif command -v pacman >/dev/null 2>&1; then
        log "Installing Arch Linux build dependencies"
        sudo pacman -Sy --needed --noconfirm base-devel ca-certificates curl \
            pkgconf python python-pip cfitsio fftw gsl
        PYTHON="$(command -v python)"
    else
        fail "Unsupported Linux distribution. Install a C toolchain, Python 3, CFITSIO, FFTW3, GSL, pkg-config, curl, and make, then rerun."
    fi
else
    fail "Unsupported operating system: $(uname -s). Use install.ps1 on Windows."
fi

command -v "$PYTHON" >/dev/null 2>&1 || fail "Python was not installed."
command -v curl >/dev/null 2>&1 || fail "curl was not installed."
command -v make >/dev/null 2>&1 || fail "make was not installed."

log "Creating the application virtual environment"
"$PYTHON" -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV/bin/python" -m pip install -r "$ROOT/requirements.txt"

log "Downloading and building glafic ${GLAFIC_VERSION}"
mkdir -p "$BUILD_ROOT"
ARCHIVE="$BUILD_ROOT/glafic-${GLAFIC_VERSION}.tar.gz"
SOURCE="$BUILD_ROOT/glafic2-${GLAFIC_COMMIT}"
if [[ ! -d "$SOURCE" ]]; then
    curl --fail --location --retry 3 \
        "https://github.com/oguri/glafic2/archive/${GLAFIC_COMMIT}.tar.gz" \
        --output "$ARCHIVE"
    tar -xzf "$ARCHIVE" -C "$BUILD_ROOT"
    SOURCE="$(find "$BUILD_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'glafic2-*' | head -n 1)"
fi

if [[ "$(uname -s)" == "Linux" ]]; then
    LIBPATH="${LIBPATH:-$(dirname "$(find /usr /lib -name 'libcfitsio.a' -print -quit 2>/dev/null)")}"
    INCPATH="${INCPATH:-$(dirname "$(find /usr /include -path '*/fitsio.h' -print -quit 2>/dev/null)")}"
fi
[[ -n "${LIBPATH:-}" && -n "${INCPATH:-}" ]] || fail "Could not locate CFITSIO headers and static libraries."
[[ -f "$INCPATH/fitsio.h" ]] || fail "Could not locate CFITSIO headers in $INCPATH."

LINK_LIBS="$(pkg-config --libs cfitsio fftw3 gsl 2>/dev/null || true)"
if [[ -n "$LINK_LIBS" ]]; then
    LINK_FLAGS="-lm $LINK_LIBS"
else
    [[ -f "$LIBPATH/libcfitsio.a" ]] || fail "Could not locate $LIBPATH/libcfitsio.a."
    [[ -f "$LIBPATH/libfftw3.a" ]] || fail "Could not locate $LIBPATH/libfftw3.a."
    [[ -f "$LIBPATH/libgsl.a" ]] || fail "Could not locate $LIBPATH/libgsl.a."
    LINK_FLAGS="-lm $LIBPATH/libcfitsio.a $LIBPATH/libfftw3.a $LIBPATH/libgsl.a $LIBPATH/libgslcblas.a"
fi

make -C "$SOURCE" clean >/dev/null 2>&1 || true
make -C "$SOURCE" -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)" \
    LIBPATH="$LIBPATH" INCPATH="$INCPATH" PYTHON="$PYTHON" LIBS="$LINK_FLAGS"
install -m 755 "$SOURCE/glafic" "$ROOT/glafic"
chmod +x "$ROOT/install.sh" "$ROOT/run.sh"

log "Verifying the installation"
"$ROOT/glafic" >/dev/null 2>&1 || true
test -x "$ROOT/glafic" || fail "glafic was not built as an executable."
test -x "$VENV/bin/python" || fail "The Python virtual environment is incomplete."

cat <<EOF

Installation complete.
Start the application with:
  $ROOT/run.sh
EOF
