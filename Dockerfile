ARG PYTHON_VER
FROM python:$PYTHON_VER
RUN apt-get -y update && apt-get install -y unzip libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev xvfb patchelf ffmpeg
RUN pip install pytest pytest-rerunfailures
ARG CACHEBUST=1
COPY . /usr/local/robo-gym/
WORKDIR /usr/local/robo-gym/
RUN pip install .
ENTRYPOINT ["/usr/local/robo-gym/bin/docker_entrypoint"]
CMD ["bash"]
