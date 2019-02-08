FROM ubuntu:16.04

WORKDIR /home

ENV DEBIAN_FRONTEND noninteractive

# Ubuntu packages + Numpy
RUN apt-get update \
     && apt-get install -y --no-install-recommends \
        apt-utils \
        build-essential \
        sudo \
        less \
        jed \
        g++  \
        git  \
        curl  \
        cmake \
        zlib1g-dev \
        libjpeg-dev \
        xvfb \
        libav-tools \
        xorg-dev \
        libboost-all-dev \
        libsdl2-dev \
        dbus \
        swig \
        python  \
        python-dev  \
        python-future  \
        python-pip  \
        python-setuptools  \
        python-wheel  \
        python-tk \
        python-opengl \
        libopenblas-base  \
        libatlas-dev  \
#        cython3  \
     && apt-get upgrade -y \
     && apt-get clean \
     && rm -rf /var/lib/apt/lists/*

# Install graphic driver
RUN apt-get install -y libgl1-mesa-dri libgl1-mesa-glx --no-install-recommends
RUN dbus-uuidgen > /etc/machine-id

# create user account
ENV USER tokudo
ENV HOME /home/${USER}
RUN export uid=1000 gid=1000 &&\
    echo "${USER}:x:${uid}:${gid}:Developer,,,:${HOME}:/bin/bash" >> /etc/passwd &&\
    echo "${USER}:x:${uid}:" >> /etc/group &&\
    echo "${USER} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers &&\
    install -d -m 0755 -o ${uid} -g ${gid} ${HOME}
WORKDIR ${HOME}

# Install python library
RUN pip install --upgrade pip
RUN pwd
RUN pip install numpy Theano opencv-python
RUN git clone https://github.com/mgbellemare/Arcade-Learning-Environment.git
RUN cd Arcade-Learning-Environment
RUN apt-get update && apt-get install libsdl1.2-dev libsdl-gfx1.2-dev libsdl-image1.2-dev cmake
RUN mkdir build && cd build
RUN cmake -DUSE_SDL=ON -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=ON ..
RUN make -j 4
RUN pip install .
RUN pip install -r https://raw.githubusercontent.com/Lasagne/Lasagne/master/requirements.txt
RUN pip install https://github.com/Lasagne/Lasagne/archive/master.zip

# X
ENV DISPLAY :0.0
VOLUME /tmp/.X11-unix
VOLUME ${HOME}
USER ${USER}

CMD [ "/bin/bash" ]
