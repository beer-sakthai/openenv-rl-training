FROM ghcr.io/openenv/browsergym-env:latest

# Install Playwright + Chromium explicitly
USER root

RUN pip install --no-cache-dir playwright && \
    playwright install chromium && \
    playwright install-deps chromium

# Expose FastAPI server port (HF Spaces requirement: 7860)
EXPOSE 7860

ENV PORT=7860
ENV BROWSERGYM_BENCHMARK=miniwob
ENV BROWSERGYM_TASK_NAME=click-test
ENV BROWSERGYM_HEADLESS=true
ENV BROWSERGYM_VIEWPORT_WIDTH=1280
ENV BROWSERGYM_VIEWPORT_HEIGHT=720

# HF Spaces runs as non-root user 1000
RUN useradd -m -u 1000 user 2>/dev/null || true && \
    chown -R 1000:1000 /home/user
USER 1000

CMD ["python", "app.py", "--port", "7860", "--host", "0.0.0.0"]
