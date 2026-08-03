FROM ghcr.io/openenv/browsergym-env:latest

# Expose FastAPI server port
EXPOSE 7860

# Override default port to 7860 (HF Spaces requirement)
ENV PORT=7860
ENV BROWSERGYM_BENCHMARK=miniwob
ENV BROWSERGYM_TASK_NAME=click-test
ENV BROWSERGYM_HEADLESS=true
ENV BROWSERGYM_VIEWPORT_WIDTH=1280
ENV BROWSERGYM_VIEWPORT_HEIGHT=720

# HF Spaces runs as non-root user 1000
RUN useradd -m -u 1000 user || true
USER 1000

CMD ["python", "app.py", "--port", "7860", "--host", "0.0.0.0"]
