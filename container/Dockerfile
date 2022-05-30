FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
# Learn how to make a multistage dockerfile
# FROM continuumio/miniconda3:latest

ENV DEBIAN_FRONTEND=noninteractive \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/compat \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64 \
    LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt update && apt upgrade -y \
    && apt install -y libpq-dev \
        build-essential git sudo \
        cmake zlib1g-dev libjpeg-dev \
        xvfb ffmpeg xorg-dev \
        libboost-all-dev libsdl2-dev \
        swig unzip zip wget \
    && rm -rf /var/lib/apt/lists/*

# ****************** Install Miniconda ************************* #
ENV PATH /opt/conda/bin:$PATH

CMD [ "/bin/bash" ]

# Leave these args here to better use the Docker build cache
ARG CONDA_VERSION=py38_4.9.2
ARG CONDA_MD5=122c8c9beb51e124ab32a0fa6426c656

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-x86_64.sh -O miniconda.sh && \
    echo "${CONDA_MD5}  miniconda.sh" > miniconda.md5 && \
    if ! md5sum --status -c miniconda.md5; then exit 1; fi && \
    mkdir -p /opt && \
    sh miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh miniconda.md5 && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy
# ******************************************* #


# Copy the environment file at volume mounted
# Conda clean (t removes cached package tarballs and y doesn't ask for confirmation)
COPY environment.yml /tmp/
RUN conda update -y -n base conda \
    && conda env create -f /tmp/environment.yml \
    && conda clean -y -t \
    && rm /tmp/environment.yml

ARG username
ARG userid

ARG home=/home/${username}
ARG workdir=${home}

RUN adduser ${username} --uid ${userid} --gecos '' --disabled-password \
    && echo "${username} ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/${username} \
    && chmod 0440 /etc/sudoers.d/${username}

USER ${username}
# Change the group and user ownership of WORKDIR to created user
RUN chown -R ${username}:${username} ${home}
WORKDIR ${workdir}

# Environment variable available for all subsequent instructions in the build stage
ENV PATH /opt/conda/envs/mo445/bin:$PATH

# Adds some alias to bashrc (alias to call more specific commands)
COPY bashrc.bash /tmp/
RUN conda init && cat /tmp/bashrc.bash >> ${home}/.bashrc \
    && echo "export PATH=\"${workdir}/docker/bin:$PATH\"" >> ${home}/.bashrc \
    && sudo rm /tmp/bashrc.bash