FROM python:3.12-slim
WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image so pods start with no download
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

COPY app.py chat.html ./

EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
