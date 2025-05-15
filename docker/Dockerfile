FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_CACHE_DIR=/pip_cache

WORKDIR /src

RUN apt-get update && apt-get install -y cron && rm -rf /var/lib/apt/lists/*

RUN mkdir -p ${PIP_CACHE_DIR}

COPY ./requirements.txt /src/requirements.txt

RUN pip install --upgrade pip && \
    pip install --cache-dir ${PIP_CACHE_DIR} -r requirements.txt && \
    rm -rf ${PIP_CACHE_DIR}

COPY ./src /src
COPY ./docker/entrypoint.sh /entrypoint.sh
COPY ./docker/cron /cron

RUN chmod +x /entrypoint.sh && \
    mkdir -p /src/model && \
    mkdir -p /src/logs && \
    echo "" >> /cron && \
    crontab /cron

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]