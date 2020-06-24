FROM python:3.6
RUN apt-get -y update && apt-get install -y unzip libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev xvfb patchelf ffmpeg

COPY . /usr/local/robo-gym/
WORKDIR /usr/local/robo-gym/
RUN pip install .
RUN pip install pytest

ENTRYPOINT ["/usr/local/robo-gym/bin/docker_entrypoint"]
CMD ["bash"]
