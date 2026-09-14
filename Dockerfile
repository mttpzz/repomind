FROM python:3.12-slim

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Run the main application as a module so absolute "src.*" imports resolve;
# ingest/query subcommand and flags are passed at `docker run` time
ENTRYPOINT ["python", "-m", "src.main"]