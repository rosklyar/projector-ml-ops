FROM huggingface/transformers-pytorch-gpu:4.22.1

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

RUN ln -s /usr/bin/python3 /usr/bin/python

ENV PYTHONPATH /app
COPY ./garbage_classifier /app/garbage_classifier 

CMD [ "bash" ]