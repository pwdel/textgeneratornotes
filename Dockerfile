# set base image
FROM tensorflow/tensorflow:latest-gpu

# set the working directory in the container
WORKDIR /usr/src/

# install system dependencies
RUN apt-get update

# install dependencies
RUN pip install --upgrade pip

# copy the content of local src to working
COPY ./requirements.txt /usr/src/

RUN pip install -r requirements.txt

COPY . /usr/src/
