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