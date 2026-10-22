FROM python:3.13-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Baixa o modelo de embedding no build para o backend não esperar no primeiro start
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

EXPOSE 8000 8001
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]
