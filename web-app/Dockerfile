FROM python:3.9
 
WORKDIR /garbage-classifier

COPY ./requirements.txt /garbage-classifier/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /garbage-classifier/requirements.txt

COPY ./source /garbage-classifier/source
 
CMD ["uvicorn", "source.main:app", "--host", "0.0.0.0", "--port", "80"]