#!/usr/bin/env bash
# Build linus's container images from a commit (docker/README.md).
#
#   docker/build.sh [gpu|cpu|all] [<commit>]       # default: gpu, HEAD
#
# `all` builds the GPU image, then the CPU image, whose build stage comes from the GPU build's
# layer cache (Dockerfile.cpu): the two take little more than the GPU image alone.
#
# The build context is `git archive <commit>`, unpacked into a fresh directory under
# $BUILD_DIR. The Dockerfiles
# are this checkout's docker/Dockerfile.{gpu,cpu}, so an older commit builds with today's
# recipe. Tags: ghcr.io/atomsforhumanity/linus:<version>-<shortcommit>, with -cuda13 for the GPU
# image; <version> is the workspace version in the commit's Cargo.toml.
#
# Environment:
#   DOCKER_BUILD_ARGS  extra `docker build` arguments, e.g. hard limits for a shared machine with
#                      the legacy builder: DOCKER_BUILDKIT=0 DOCKER_BUILD_ARGS="--cpuset-cpus=16-31 --memory=48g"
#   BUILD_DIR          where the context is unpacked (default: $TMPDIR or /tmp; about 80 MB)
#   IMAGE              repository (default ghcr.io/atomsforhumanity/linus)
#   PUSH=1             push the tags after building (needs `docker login ghcr.io`)
#   MIRRORS            with PUSH=1, more registries to push the same image to, as
#                      <registry>/linus:<tag> (space-separated, e.g. a cluster's LAN registry)
#   KEEP_CONTEXT=1     keep the unpacked context
set -euo pipefail

KIND=${1:-gpu}
COMMIT=${2:-HEAD}
IMAGE=${IMAGE:-ghcr.io/atomsforhumanity/linus}
case "$KIND" in
    gpu | cpu) KINDS=("$KIND") ;;
    all) KINDS=(gpu cpu) ;;
    *) echo "usage: $0 [gpu|cpu|all] [<commit>]" >&2; exit 2 ;;
esac

DOCKER_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(git -C "$DOCKER_DIR" rev-parse --show-toplevel)

# The CPU image's build stage starts with the GPU image's, line for line, so that Docker's layer
# cache shares it (Dockerfile.cpu); a difference would silently build everything twice.
build_stage() { awk '/^FROM / { n++ } n == 1' "$1" | sed '/^$/d'; }
gpu_stage=$(build_stage "$DOCKER_DIR/Dockerfile.gpu")
if [ "$(build_stage "$DOCKER_DIR/Dockerfile.cpu" | head -n "$(wc -l <<< "$gpu_stage")")" != "$gpu_stage" ]; then
    echo "docker/Dockerfile.cpu's build stage no longer starts with Dockerfile.gpu's: keep them in step" >&2
    exit 1
fi
SHA=$(git -C "$REPO" rev-parse --verify "$COMMIT^{commit}")
SHORT=$(git -C "$REPO" rev-parse --short=7 "$SHA")
VERSION=$(git -C "$REPO" show "$SHA:Cargo.toml" \
    | awk '/^\[workspace.package\]/ { s = 1; next } /^\[/ { s = 0 } s && /^version *=/ { gsub(/[" ]/, "", $0); sub(/^version=/, "", $0); print; exit }')
[ -n "$VERSION" ] || { echo "no [workspace.package] version in $SHORT:Cargo.toml" >&2; exit 1; }

CTX=$(mktemp -d "${BUILD_DIR:-${TMPDIR:-/tmp}}/linus-docker-$SHORT.XXXXXX")
cleanup() { [ -n "${KEEP_CONTEXT:-}" ] || rm -rf "$CTX"; }
trap cleanup EXIT
git -C "$REPO" archive --format=tar "$SHA" | tar -x -C "$CTX"
echo "context: $CTX ($SHORT, version $VERSION)"

for kind in "${KINDS[@]}"; do
    tag="$IMAGE:$VERSION-$SHORT"
    [ "$kind" = gpu ] && tag="$tag-cuda13"
    t0=$(date +%s)
    # shellcheck disable=SC2086  # DOCKER_BUILD_ARGS is a list of arguments
    docker build ${DOCKER_BUILD_ARGS:-} -f "$DOCKER_DIR/Dockerfile.$kind" \
        --label org.opencontainers.image.revision="$SHA" \
        --label org.opencontainers.image.version="$VERSION" \
        --label org.opencontainers.image.source=https://github.com/atomsforhumanity/linus \
        -t "$tag" "$CTX"
    echo "built $tag in $(( $(date +%s) - t0 )) s"
    if [ -n "${PUSH:-}" ]; then
        docker push "$tag"
        for m in ${MIRRORS:-}; do
            docker tag "$tag" "$m/linus:${tag##*:}"
            docker push "$m/linus:${tag##*:}"
        done
    fi
done
