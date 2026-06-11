FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Pake Gunicorn buat produksi, jangan pake flask run
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
