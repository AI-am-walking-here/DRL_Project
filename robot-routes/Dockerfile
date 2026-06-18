FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libegl1 libgl1 libglew2.2 libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY data ./data
COPY calibration ./calibration

ENV MUJOCO_GL=egl
ENV PYTHONPATH=src
ENV HDF5_USE_FILE_LOCKING=FALSE

RUN pip install --no-cache-dir -e ".[dev]"

CMD ["make", "check"]
