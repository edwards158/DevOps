FROM continuumio/anaconda3:4.4.0
MAINTAINER REDWARDS
COPY ./flask_demo ./
EXPOSE 5000
WORKDIR ./
RUN pip install -r requirements.txt
CMD python flask_predict_api.py