FROM python:3.9-slim-buster

RUN useradd --create-home appuser

WORKDIR /home/appuser/labelling_bot

USER appuser

COPY ./requirements.txt /home/appuser/labelling_bot/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /home/appuser/labelling_bot/requirements.txt

COPY ./labelling_bot /home/appuser/labelling_bot

CMD ["python", "main.py"]