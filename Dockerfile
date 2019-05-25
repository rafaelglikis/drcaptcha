FROM tensorflow/tensorflow:latest-py3

MAINTAINER Rafael Glikis <rafaelglikis@gmail.com>

ENV NAME DrCaptcha

WORKDIR /install

RUN pip install scipy
RUN pip install tqdm
RUN pip install numpy
RUN pip install pandas
RUN pip install tensorflow
RUN pip install Django
RUN pip install matplotlib
RUN pip install seaborn
RUN pip install Pillow
RUN pip install django-picklefield
RUN pip install scikit_learn
RUN pip install pydot
RUN pip install h5py

RUN apt update -y
RUN apt install -y graphviz python3-tk
RUN mkdir -p ~/.config/matplotlib/
RUN echo "backend: Agg" > ~/.config/matplotlib/matplotlibrc

EXPOSE 8000

WORKDIR /app
