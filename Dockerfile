FROM python:3.9
ENV PYTHONUNBUFFERED=1
WORKDIR /model_backend

RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx

COPY  . .
RUN pip3 install -r requirements.txt
RUN pip3 install gunicorn