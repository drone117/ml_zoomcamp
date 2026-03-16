FROM python:3.13.12-trixie
LABEL authors="persik"

WORKDIR /usr/src/app
ENV PATH="/usr/src/app/.venv/bin:$PATH"
COPY ["pyproject.toml", "uv.lock", "predict.py", "model_C=1.0.bin", "request.json", "./"]
RUN pip install --no-cache-dir uv &&  \
    uv sync --locked
ENV UV_NO_DEV=1
EXPOSE 8000
ENTRYPOINT ["hypercorn","--bind", "0.0.0.0:8000", "predict:app"]
