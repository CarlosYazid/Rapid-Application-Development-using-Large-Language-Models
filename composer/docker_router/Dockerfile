FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi==0.135.1 uvicorn==0.41.0 docker==7.1.0 && pip check
COPY docker_router.py .
CMD ["uvicorn", "docker_router:app", "--host", "0.0.0.0", "--port", "8070"]
