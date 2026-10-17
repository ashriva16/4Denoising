# 4Denoising container: identical environment on a workstation (Docker) and an
# HPC cluster (Apptainer/Singularity). See docs/container.md.
#
#   docker build -t 4denoising:0.1.0 --build-arg GIT_COMMIT=$(git rev-parse HEAD) .
#
# The base image fixes Python, PyTorch 2.9.1 and CUDA 12.8 (needs an NVIDIA
# driver >= 570). For an older driver, build on a lower CUDA tag via BASE_IMAGE.
ARG BASE_IMAGE=pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}

ARG GIT_COMMIT=unknown
ARG IMAGE_TAG=4denoising
ENV FOURDENOISING_COMMIT=${GIT_COMMIT} \
    FOURDENOISING_IMAGE=${IMAGE_TAG} \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/4denoising \
    MPLBACKEND=Agg

COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

WORKDIR /opt/4denoising
COPY VERSION LICENSE README.md pyproject.toml ./
COPY core core
COPY scripts scripts
COPY utils utils
COPY configs configs

# fail the build early if the package does not import
RUN python -c "import core.readers, core.stream, core.defects, core.checkpoints, core.inference, core.models"

# Data, configs and results are mounted at run time; paths in configs refer to the
# container side (e.g. /data, /runs).
WORKDIR /work
ENTRYPOINT ["python", "-m"]
CMD ["scripts.infer", "--help"]
