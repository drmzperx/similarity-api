ARG PYTHON_VERSION=3.11.4
ARG SHA256=c46b0ae5728c2247b99903098ade3176a58e274d9c7d2efeaaab3e0621a53935

# BUILD stage
FROM python:${PYTHON_VERSION}-slim@sha256:${SHA256} AS build

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt


# PROD stage
FROM python:${PYTHON_VERSION}-slim-buster@sha256:${SHA256} AS prod

WORKDIR /app
COPY ./src/ /app

COPY --from=build /app/wheels /wheels
COPY --from=build /app/requirements.txt .

RUN pip install --no-cache /wheels/*

CMD uvicorn app:app --host ${BACKEND_HOST} --port ${PORT}



# DEV stage
FROM python:${PYTHON_VERSION}-slim-buster@sha256:${SHA256} AS dev

WORKDIR /app

COPY --from=build /app/wheels /wheels
COPY --from=build /app/requirements.txt .

RUN pip install --no-cache /wheels/*
RUN pip install debugpy

CMD python -Xfrozen_modules=off -m debugpy --wait-for-client --listen 0.0.0.0:9226 -m uvicorn app:app --reload --host ${BACKEND_HOST} --port ${PORT}