FROM python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.lock /app/requirements.lock

RUN python -m pip install --require-hashes --requirement /app/requirements.lock

COPY agent.py law_store.py /app/

CMD ["python", "-c", "import time; time.sleep(10**9)"]
