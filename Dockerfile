FROM ubuntu:20.04
RUN apt-get update

# Install some required dependencies.
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get update && apt-get install -y \
    zlib1g zlib1g-dev libsqlite3-dev libssl-dev git wget cmake build-essential \
    ninja-build libprotobuf-dev protobuf-compiler lsb-release \
    software-properties-common dh-autoreconf && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://apt.llvm.org/llvm.sh && \
    bash llvm.sh 13 && \
    ln -s /usr/bin/clang-13 /usr/bin/clang && \
    ln -s /usr/bin/clang++-13 /usr/bin/clang++

ENV CC /usr/bin/clang
ENV CXX /usr/bin/clang++

# S6 requires CPython to have been built with frame-pointers enabled.
ENV CFLAGS=-fno-omit-frame-pointer
ENV CXXFLAGS=-fno-omit-frame-pointer

# Install python 3.7 from source.
ARG PYTHON_SRC="/open_s6/python_src"
ENV PYTHON_INSTALL="/open_s6/python_3_7"
WORKDIR /open_s6
RUN mkdir -p $PYTHON_INSTALL $PYTHON_SRC && \
    git clone https://github.com/python/cpython.git $PYTHON_SRC && \
    cd $PYTHON_SRC  && \
    git checkout 3.7 && \
    ./configure -enable-shared \
    --prefix=$PYTHON_INSTALL LDFLAGS=-Wl,-rpath=$PYTHON_INSTALL/lib && \
    make -j install && cd /open_s6 && rm -rf $PYTHON_SRC

ENV PATH="${PYTHON_INSTALL}/bin:${PATH}"
RUN python3.7 -m pip install --upgrade pip absl-py numpy ipython jupyterlab

# Copy all local files to the container.
RUN mkdir s6
COPY . s6/

# Build the project.
WORKDIR /open_s6/s6
RUN ./build.sh `pwd`/src `pwd`/build -GNinja -DPython_ROOT_DIR=$PYTHON_INSTALL

# Setup S6's python module layout.
ENV PYTHONPATH="/open_s6/python_modules/"
RUN bash ./setup_python_module.sh `pwd`/src `pwd`/build "${PYTHONPATH}"
