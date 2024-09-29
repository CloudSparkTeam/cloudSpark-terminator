# Usa uma imagem base de Python slim para manter o contêiner leve
FROM python:3.9-slim

# Define o diretório de trabalho dentro do contêiner
WORKDIR /app

# Copia o arquivo de dependências para dentro do contêiner
COPY yolov8-cloud-detection/src/api/requirements.txt .

# Instala as dependências necessárias
RUN pip install --no-cache-dir -r requirements.txt

# Copia o script principal e o modelo para dentro do contêiner
COPY yolov8-cloud-detection/src/api/main.py .
COPY yolov8-cloud-detection/runs/train/cloud-detection/weights/best.pt ./weights/

# Expõe a porta 8000 para acessar o FastAPI
EXPOSE 8000

# Comando que será rodado quando o contêiner iniciar
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
