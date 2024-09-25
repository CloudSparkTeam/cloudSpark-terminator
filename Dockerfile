# Usa uma imagem base de Python slim para manter o contêiner leve
FROM python:3.9-slim

# Instala bibliotecas necessárias para o OpenCV e outras dependências do sistema
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Define o diretório de trabalho dentro do contêiner
WORKDIR /app

# Copia o arquivo de dependências para dentro do contêiner
COPY yolov8-cloud-detection/src/api/requirements.txt .

# Instala as dependências necessárias
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante do código para o contêiner
COPY yolov8-cloud-detection/ .

# Expõe a porta 8000 para acessar o FastAPI
EXPOSE 8000

# Comando que será rodado quando o contêiner iniciar
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
