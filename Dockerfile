# Runs the pipeline (synthetic feeder + interactive map), the comparison with the real feeder and the automated tests
# without installing anything on the host.
#
#   docker build -t gridstudio-enel .
#   docker run --rm -v <ENEL data folder>:/data:ro -v <output folder>:/output gridstudio-enel            (= pipeline)
#
# Or with docker compose: see DOCKER.md. The ENEL data is NOT part of the image: it is provided by ENEL and mounted at /data.
FROM python:3.12-slim

# Image metadata (shown by "docker inspect" and on Docker Hub)
LABEL org.opencontainers.image.title="GridStudio ENEL pipeline" \
      org.opencontainers.image.description="Reference-based synthesis and validation of a medium-voltage distribution feeder (Schweitzer et al., 2017)" \
      org.opencontainers.image.source="https://github.com/Adamlahmar1/GridStudio" \
      org.opencontainers.image.authors="Adam Lahmar"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    PYTHONPATH=/app \
    ENEL_DATA_DIR=/data \
    OUTPUT_DIR=/output

WORKDIR /app

# Dependencies first: this layer is cached, so a code change does not re-download every package.
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# The source is copied instead of "pip install ." because pyproject.toml refers to
# README.md while the file is called readme.md (fine on Windows, fails on Linux).
COPY powergrid_synth ./powergrid_synth
COPY feeder_pipeline ./feeder_pipeline
COPY pipeline.py compare_real_fake.py gridstudio.py ./
COPY tests ./tests

VOLUME ["/data", "/output"]

# "docker run ... gridstudio-enel" = pipeline; the first argument may be a command (compare, test, help);
# anything else is passed on to the pipeline, e.g. "docker run ... gridstudio-enel --seed 7".
ENTRYPOINT ["python", "gridstudio.py"]
