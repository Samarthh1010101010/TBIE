# TBIE inference image.
#
# STATUS: paths and structure verified; the image has NOT been built end to end
# (no Docker daemon available on the authoring machine). Treat the first build
# as unproven. `make docker-build` runs it.
#
# Pinned to Python 3.12 because the frozen models are pickled under 3.12 and
# will not deserialise on 3.14+. This replaces the previous approach of having
# pipeline.py shell out to `pip install` at runtime.
#
# Build:
#   docker build -t tbie .
#
# Run (mount the raw data in, and the output directory out):
#   docker run --rm \
#     -v "$(pwd)/data/train:/app/data/train:ro" \
#     -v "$(pwd)/outputs:/app/outputs" \
#     tbie --data_dir ./data/train/ \
#          --observation_date 2025-12-31 \
#          --output_dir ./outputs/

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUTF8=1

WORKDIR /app

# Dependencies first so the layer caches across source edits.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Source and frozen model artifacts. The large regenerable parquet directories
# (features/, snapshots/, states/, segments/*.parquet) are excluded via
# .dockerignore — the pipeline rebuilds what it needs from raw data.
COPY pipeline.py ./
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY models/ ./models/
COPY segments/segment_model.pkl ./segments/
COPY segments/segment_definitions.json ./segments/

# Fail the build if the shipped bundles do not satisfy the inference contract.
RUN python scripts/check_model_contract.py

ENTRYPOINT ["python", "pipeline.py"]
CMD ["--help"]
