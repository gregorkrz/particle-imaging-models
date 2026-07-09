# syntax=docker/dockerfile:1

ARG TORCH_VERSION=2.5.0
ARG CUDA_VERSION=12.4
ARG CUDNN_VERSION=9
FROM pytorch/pytorch:${TORCH_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel

COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /uvx /bin/

ARG TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0"
ENV CUDA_HOME=/usr/local/cuda \
    FORCE_CUDA=1 \
    MAX_JOBS=1 \
    PATH=/opt/pimm/.venv/bin:$PATH \
    PYTHONNOUSERSITE=1 \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/pimm/.venv

RUN rm -f /etc/apt/sources.list.d/*.list
COPY .github/docker/common /opt/pimm/.github/docker/common
RUN bash /opt/pimm/.github/docker/common/install_system.sh

WORKDIR /opt/pimm/src
COPY pyproject.toml uv.lock .python-version ./
COPY libs/ ./libs/
COPY pimm/ ./pimm/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --all-extras --group docs --locked

COPY --exclude=libs --exclude=pimm --exclude=pyproject.toml --exclude=uv.lock \
    --exclude=.python-version . /opt/pimm/src

COPY .github/docker/entrypoint.sh /opt/pimm/entrypoint.sh
RUN chmod +x /opt/pimm/entrypoint.sh
ENTRYPOINT ["/opt/pimm/entrypoint.sh"]
CMD ["bash"]
