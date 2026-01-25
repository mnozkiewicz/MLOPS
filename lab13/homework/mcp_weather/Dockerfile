FROM python:3.12-slim

WORKDIR /app

RUN pip install fastmcp requests

COPY api.py .

EXPOSE 8001

CMD ["python", "api.py"]