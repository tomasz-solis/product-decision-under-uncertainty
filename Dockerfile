# Reproducible container for the Streamlit exploration app.
# Uses the committed uv.lock so the image matches CI exactly.
FROM python:3.11-slim

# Bring in the uv binary from the official distroless image.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install dependencies first for better layer caching. --frozen fails the build
# if uv.lock is out of sync with pyproject.toml, which is the behaviour we want.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy the application source.
COPY . .

EXPOSE 8501

# Streamlit must bind 0.0.0.0 to be reachable from outside the container.
CMD ["uv", "run", "--no-dev", "streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
