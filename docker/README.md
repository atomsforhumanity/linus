# Container images

Two images of linus for cluster jobs, built from a commit of this repository:

| image | Dockerfile | base | contents |
|---|---|---|---|
| `ghcr.io/atomsforhumanity/linus:<version>-<commit>-cuda13` | `Dockerfile.gpu` | `nvidia/cuda:13.0.3-runtime-ubuntu24.04` | the `linus` command and the Python package (`afh-linus-cuda13` with qccompute and qcdata, in the venv `/opt/venv`, first on `PATH`), both with the GPU backend (`gpu` feature) |
| `ghcr.io/atomsforhumanity/linus:<version>-<commit>` | `Dockerfile.cpu` | `python:3.12.12-slim-bookworm` | the same without the GPU backend (`afh-linus`), in the image's own Python |

The Python package comes with the qc* stack's release: qcdata 0.24.0 and qccompute 0.23.0.
Both have the basis set library in `/usr/local/share/linus/basis` (`LINUS_BASIS_DIR`), run as
root, and check at build time that `linus` computes a water energy on the CPU. The GPU image
needs an NVIDIA driver with CUDA 13 support (580 or newer) on the host; its CUDA 13.0 runtime
libraries (NVRTC, cuBLAS, cuSOLVER) are what linus loads when a calculation first asks for the
GPU (docs/gpu.md). The build checks only the CPU path: check the GPU on a GPU, e.g.

```bash
docker run --rm --gpus 1 <image> sh -c 'printf "3\n\nO 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692\n" > /tmp/w.xyz \
  && linus /tmp/w.xyz --method b3lyp --basis def2-svp --jk df --backend gpu --json' | python3 -c \
  'import json, sys; r = json.load(sys.stdin); print(r["energy"], r["resolved"]["backend"]["jk"], r["resolved"]["backend"]["device"])'
```

## Building

```bash
docker/build.sh all <commit>          # the GPU image, then the CPU image; gpu or cpu builds one
PUSH=1 docker/build.sh all <commit>   # and push both tags (needs `docker login ghcr.io`)
PUSH=1 MIRRORS=<registry> docker/build.sh all <commit>   # also as <registry>/linus:<tag>, e.g. a cluster's LAN registry
```

- The build context is `git archive <commit>`, unpacked under `$BUILD_DIR` (default
  `$TMPDIR`). The Dockerfiles are those of the checkout that runs the script.
- Tags: `<version>-<shortcommit>` (the workspace version in that commit's Cargo.toml), with
  `-cuda13` for the GPU image. Use a new commit for every change: cluster nodes cache images by
  tag.
- One shared build stage: `Dockerfile.cpu` starts with `Dockerfile.gpu`'s build stage line for
  line, so after the GPU image Docker's layer cache supplies it and only the CPU variant is
  compiled (the dependencies and libxc are not built again). `build.sh` refuses to run if the
  two stop matching.
- The builds compile with `--profile dist` (the release profile, stripped; `codegen-units = 1`
  with thin LTO, so the `linus` crate itself compiles on one core for several minutes).
- Docker builds run in the Docker daemon, outside the caller's CPU and memory limits. On a
  shared machine, cap them with the legacy builder:
  `DOCKER_BUILDKIT=0 DOCKER_BUILD_ARGS="--cpuset-cpus=16-31 --memory=48g --memory-swap=48g" docker/build.sh all <commit>`.

## Using the images on a cluster

- Put linus's GPU kernel cache on storage the pods share (`backend.kernel_cache`, or its
  default `LINUS_GPU_CACHE`): the image's home
  directory is empty in every new container, and the first calculation of a kind compiles its
  kernels (up to a minute). docs/gpu.md, "Containers and cluster jobs".
- Set the thread count per calculation (`--threads`, or the `threads` keyword) from the pod's
  CPU limit; the container sees the whole machine's cores.
