FROM python:3.12-slim

WORKDIR /app
COPY requirements-chat.txt .
RUN pip install --no-cache-dir -r requirements-chat.txt
COPY academic_chat_api.py .
COPY GBAZP-CHAT ./GBAZP-CHAT

ENV PORT=5000
EXPOSE 5000
CMD ["python", "academic_chat_api.py"]
