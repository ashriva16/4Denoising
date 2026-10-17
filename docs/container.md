# Running 4Denoising in a container

One image gives the same environment on a workstation (Docker) and on an HPC
cluster (Apptainer/Singularity, which runs Docker images without root). The image
holds the code and its dependencies only; data, configs, weights and results are
mounted from outside.

## Build (once per code version)

```bash
docker build -t 4denoising:0.1.0 \
    --build-arg GIT_COMMIT=$(git rev-parse HEAD) \
    --build-arg IMAGE_TAG=4denoising:0.1.0 .
```

The commit and tag are recorded in the provenance of every checkpoint and every
denoised file, so a result names the exact image that produced it. The base image
(`pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime`) needs an NVIDIA driver >= 570;
check with `nvidia-smi`. For an older driver, add
`--build-arg BASE_IMAGE=pytorch/pytorch:<version>-cuda<lower>-cudnn9-runtime`.

## Workstation (Windows: Docker Desktop with the WSL 2 backend)

GPU support needs a current NVIDIA driver on Windows; nothing is installed inside WSL.

```bat
docker run --rm --gpus all ^
    -v "E:\data":/data:ro ^
    -v "C:\runs":/runs ^
    4denoising:0.1.0 scripts.infer --config /runs/cfg.yml --checkpoint /runs/w.pth ^
    --source all --output /runs/denoised/ --amp
```

Paths inside the config refer to the container side (`/data/...`, `/runs/...`).
The image's entry point is `python -m`, so any script runs the same way:
`scripts.train`, `scripts.make_defect_mask`, `scripts.infer`.

## HPC (Apptainer)

Most clusters do not allow Docker but run its images with Apptainer. Either push the
image to a registry and pull it on the cluster,

```bash
docker tag 4denoising:0.1.0 ghcr.io/<user>/4denoising:0.1.0
docker push ghcr.io/<user>/4denoising:0.1.0
# on the cluster
apptainer pull 4denoising_0.1.0.sif docker://ghcr.io/<user>/4denoising:0.1.0
```

or copy it as a file: `docker save 4denoising:0.1.0 -o 4denoising.tar`, then on the
cluster `apptainer build 4denoising_0.1.0.sif docker-archive://4denoising.tar`.

Run with `--nv` for the GPU and `--bind` for the data:

```bash
apptainer run --nv --bind /scratch/$USER/data:/data,/scratch/$USER/runs:/runs \
    4denoising_0.1.0.sif scripts.infer --config /runs/cfg.yml --checkpoint /runs/w.pth \
    --source all --output /runs/denoised/ --amp
```

A batch job only wraps this line (e.g. in an `sbatch` script requesting one GPU).
Outputs are resumable, so a job that hits its time limit can simply be submitted again.

## Same environment, not bit-identical numbers

The image fixes the software. Results from different GPU generations can still
differ in the last digits (convolution algorithms, mixed precision); the GPU name
and the precision used are in the provenance of every output.
