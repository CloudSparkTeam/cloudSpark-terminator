# Use uma imagem base do Python com suporte ao PyTorch
FROM python:3.9

# Defina o diretório de trabalho no container
WORKDIR /app

# Copie os arquivos necessários para a aplicação
COPY yolov8-cloud-detection/src/api/main.py ./
COPY yolov8-cloud-detection/src/api/requirements.txt ./
COPY yolov8-cloud-detection/notebooks/model_training_Unet.ipynb ./unet_model.pth

# Instale as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Exponha a porta em que o Flask irá rodar
EXPOSE 8000

# Comando para iniciar a aplicação Flask
CMD ["python", "main.py"]
