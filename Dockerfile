# Runs the ENEL pipeline (pipeline.py) without installing anything on the host.
#
#   docker build -t gridstudio-enel .
#   docker run --rm -v <ENEL data folder>:/data:ro -v <output folder>:/output gridstudio-enel
#
# The ENEL data is NOT part of the image: it is provided by ENEL and mounted at /data.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    PYTHONPATH=/app \
    ENEL_DATA_DIR=/data \
    OUTPUT_DIR=/output

WORKDIR /app

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# The source is copied instead of "pip install ." because pyproject.toml refers to
# README.md while the file is called readme.md (fine on Windows, fails on Linux).
COPY powergrid_synth ./powergrid_synth
COPY pipeline.py .

VOLUME ["/data", "/output"]

# Extra arguments are passed on to the script, e.g. "docker run ... gridstudio-enel --seed 7"
ENTRYPOINT ["python", "pipeline.py"]
