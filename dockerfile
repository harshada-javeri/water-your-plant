FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/ ./scripts/
COPY data/ ./data/
COPY model.pkl ./

EXPOSE 9696

CMD ["python", "scripts/predict.py"]
