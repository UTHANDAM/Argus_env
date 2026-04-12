FROM python:3.11-slim

# Set environment variables for python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create a non-root user for security
RUN useradd --create-home appuser
WORKDIR /home/appuser/app

# Copy application files and set ownership
COPY --chown=appuser:appuser . .

# Install dependencies
RUN pip install --no-cache-dir -e .

# Switch to the non-root user for runtime
USER appuser

# Expose the port the app runs on
EXPOSE 7860

# Add a healthcheck to ensure the service is running
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

# Run the application
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]