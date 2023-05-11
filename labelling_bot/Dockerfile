FROM python:3.9-slim-buster

WORKDIR /labelling_bot

COPY ./requirements.txt /labelling_bot/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /labelling_bot/requirements.txt

COPY ./labelling_bot /labelling_bot/labelling_bot

CMD ["python", "labelling_bot/main.py"]