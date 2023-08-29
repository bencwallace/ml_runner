FROM nvidia/cuda:11.7.1-base-ubuntu22.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
RUN apt-get update --fix-missing \
    && apt-get install -y curl unzip  wget bzip2 ca-certificates libxrender1 mercurial subversion libgdal-dev libyaml-dev \
    && apt-get install -y sudo \
    && apt-get clean

ENV PATH="/opt/miniconda3/bin:${PATH}"
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_23.1.0-1-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/miniconda3 && \
    rm ~/miniconda.sh && \
    conda clean -tipy && \
    sudo ln -s /opt/miniconda3/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find /opt/miniconda3/ -follow -type f -name '*.a' -delete && \
    find /opt/miniconda3/ -follow -type f -name '*.js.map' -delete && \
    conda clean -afy

RUN pip install poetry
