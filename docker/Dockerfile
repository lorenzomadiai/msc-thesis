FROM pytorch/pytorch:2.5.0-cuda12.1-cudnn9-devel
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV MUJOCO_GL=osmesa

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt-get update

# Opencv dependency
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        build-essential \
        cmake \
        g++ \
        vim \
    patchelf \
        libjpeg-dev \
    libgl1-mesa-dev \
    libglfw3 \
    libglew-dev \
    libosmesa6-dev \
        librdmacm1 \
        libibverbs1 \
        ibverbs-providers \
        curl

# Optionally, install an alternative version of Python (example: Python 3.11)
RUN apt-get update && apt-get install -y python3.11 python3.11-distutils python3.11-venv && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

# Upgrade pip for Python 3.11
RUN python3 -m ensurepip && python3 -m pip install --upgrade pip
# pip install opencv-python-headless


# Install ssh server, used by PyCharm for remote execution
RUN apt-get update && apt-get install -y openssh-server
RUN mkdir /var/run/sshd
RUN exec > /etc/ssh/sshd_config &&\
    echo AddressFamily inet &&\
    echo ListenAddress 0.0.0.0 &&\
    echo PermitRootLogin yes &&\
    echo PermitUserEnvironment yes &&\
    echo Subsystem sftp /usr/lib/openssh/sftp-server
EXPOSE 10710

# Install requirements
COPY ./docker/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# git
RUN apt-get update && apt-get install -y git







