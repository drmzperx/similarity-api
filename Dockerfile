#Builder base
FROM python:3.13-slim AS builder
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV APP_HOME=/home/app
RUN mkdir -p $APP_HOME
WORKDIR $APP_HOME

#RUN apk update && apk add --no-cache bash
ADD requirements.txt $APP_HOME
RUN pip install -r $APP_HOME/requirements.txt
#Because of sentence-trasformers install the cude version of pyTourch
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install huggingface_hub sentence-transformers==3.3.1 --no-deps

#Production base
FROM python:3.13-slim

COPY --from=builder . .

COPY ./ $APP_HOME
#--reload only dev mode
#CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "5000", "--workers", "1"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000", "--workers", "1"]
