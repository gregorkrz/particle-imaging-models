# syntax=docker/dockerfile:1
#
# pimm image. Shares pins + install steps with the local conda path (install.sh)
# via .github/docker/common/*. Keep the FROM tag below in sync with .github/docker/common/versions.sh.

ARG TORCH_VERSION=2.5.0
ARG CUDA_VERSION=12.4
ARG CUDNN_VERSION=9
FROM pytorch/pytorch:${TORCH_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel

# Prevent bind-mounted ~/.local pip packages from shadowing container packages.
ENV PYTHONNOUSERSITE=1
RUN rm -f /etc/apt/sources.list.d/*.list

COPY .github/docker/common /opt/pimm/.github/docker/common
COPY .github/docker/requirements /opt/pimm/.github/docker/requirements

# System toolchain. The base image already provides torch+CUDA, so we do NOT run
# install_torch.sh here.
RUN bash /opt/pimm/.github/docker/common/install_system.sh
RUN bash /opt/pimm/.github/docker/common/install_python_deps.sh
RUN bash /opt/pimm/.github/docker/common/install_pyg.sh
RUN bash /opt/pimm/.github/docker/common/install_flash_attn.sh

# Local CUDA extensions.
COPY libs/ /opt/pimm/libs/
RUN PIMM_LIBS=/opt/pimm/libs bash /opt/pimm/.github/docker/common/install_cuda_exts.sh \
 && rm -rf /opt/pimm/libs

# Install pimm itself as editable source inside the image.
COPY . /opt/pimm/src
WORKDIR /opt/pimm/src
RUN pip install --no-cache-dir -e .

# Entrypoint neutralizes conda/mamba init blocks in a bind-mounted ~/.bashrc.
COPY .github/docker/entrypoint.sh /opt/pimm/entrypoint.sh
RUN chmod +x /opt/pimm/entrypoint.sh
ENTRYPOINT ["/opt/pimm/entrypoint.sh"]
CMD ["bash"]
