FROM python:3.9.12

RUN pip install mlflow==1.26.1
RUN pip install sklearn

COPY scripts/serving_entrypoint.sh /serving_entrypoint.sh

COPY ../artifacts /artifacts

ENTRYPOINT [ "/bin/bash", "/serving_entrypoint.sh" ]