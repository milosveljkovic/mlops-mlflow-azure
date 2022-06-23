FROM python:3-slim
LABEL maintainer="Alexander Thamm GmbH <contact@alexanderthamm.com>"
ARG MLFLOW_VERSION=1.23.1
ARG protobuf=3.20.1

WORKDIR /mlflow/
RUN pip install mlflow==$MLFLOW_VERSION
RUN pip install protobuf==$protobuf
EXPOSE 5000

RUN chmod 777 -R /mlflow

ENV BACKEND_URI sqlite:////mlflow/mlflow.db
ENV ARTIFACT_ROOT /mlflow/artifacts

CMD mlflow server --backend-store-uri  sqlite:////mlflow/mlflow.db --default-artifact-root /mlflow/artifacts --host 0.0.0.0 --port 5000
