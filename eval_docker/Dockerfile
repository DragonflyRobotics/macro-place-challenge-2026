FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel
LABEL maintainer="Yibo Lin <yibolin@pku.edu.cn>"

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
        && apt-get update \
        && apt-get install -y \
            wget \
            flex \
            libcairo2-dev \
            libboost-all-dev

# install newer bison (3.3+ required)
RUN conda install -y -c conda-forge bison 

ADD https://github.com/Kitware/CMake/releases/download/v3.25.1/cmake-3.25.1-linux-x86_64.sh /cmake-3.25.1-linux-x86_64.sh
RUN mkdir /opt/cmake \
        && sh /cmake-3.25.1-linux-x86_64.sh --prefix=/opt/cmake --skip-license \
        && ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake \
        && cmake --version


RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /challenge
COPY macro_place/ /challenge/macro_place/
COPY pyproject.toml /challenge/pyproject.toml
COPY requirements.txt /challenge/requirements.txt
COPY external/MacroPlacement/Testcases/ICCAD04/ /challenge/external/MacroPlacement/Testcases/ICCAD04/
COPY external/MacroPlacement/CodeElements/Plc_client/ /challenge/external/MacroPlacement/CodeElements/Plc_client/
COPY submissions/examples/ /challenge/submissions/examples/
RUN pip install --no-cache-dir --no-build-isolation . && \
    pip install --no-cache-dir scipy tqdm

RUN pip install --upgrade pip==21.3.1 setuptools==59.5.0 wheel
RUN pip install --no-cache-dir --only-binary=:all: \
    scipy==1.5.4 \
    numba==0.53.1
RUN pip install --no-cache-dir --no-build-isolation \
    cvxpy==1.1.18 dccp

RUN pip install --upgrade \
        numpy==1.20 \
        pyunpack>=0.1.2 \
        patool>=1.12 \
        matplotlib>=2.2.2 \
        cairocffi>=0.9.0 \
        pkgconfig>=1.4.0 \
        setuptools>=39.1.0 \
        scipy>=1.1.0 \
        absl-py
COPY submission_eval/macroformer/ /challenge/submissions/macroformer/
COPY dream/ /challenge/dream/
# RUN git clone --recursive https://github.com/gradinnovate-labs/DREAMPlace-2.x.git /DREAMPlace
COPY dreamplace/ /DREAMPlace/install

ENTRYPOINT ["python", "-m", "macro_place.evaluate"]
