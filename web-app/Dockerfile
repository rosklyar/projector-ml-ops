FROM python:3.9-slim-buster as base

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

RUN ln -s /usr/bin/python3 /usr/bin/python

ENV PYTHONPATH /app
COPY ./serving /app/serving 

CMD [ "bash" ]

# Streamlit UI docker image
FROM base AS uwg-app-streamlit
CMD streamlit run --server.address 0.0.0.0 --server.port 8080 serving/ui_app.py


# Fast API docker image
FROM base AS uwg-app-fastapi
CMD uvicorn --host 0.0.0.0 --port 8080 --workers 4 serving.fast_api:app

FROM base AS uwg-app-seldon

# Port for GRPC
EXPOSE 5000
# Port for REST
EXPOSE 9000

ENV MODEL_NAME SeldonAPI
ENV SERVICE_TYPE MODEL
# COPY /app/serving/seldon_api.py /app/SeldonAPI.py
COPY serving/seldon_api.py /app/SeldonAPI.py

RUN chown -R 8888 /app
RUN mkdir /.cache
RUN chmod 777 /.cache
RUN mkdir /.config
RUN chmod 777 /.config

CMD exec seldon-core-microservice $MODEL_NAME --service-type $SERVICE_TYPE

FROM base AS uwg-app-kserve

ENTRYPOINT ["python", "serving/kserve_api.py"]