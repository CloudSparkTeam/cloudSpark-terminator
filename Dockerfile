# Use uma imagem base do Python com suporte ao PyTorch
FROM python:3.9

# Defina o diretório de trabalho no container
WORKDIR /app

# Copie os arquivos necessários para a aplicação
COPY main.py .
COPY unet_model.pth .
COPY requirements.txt .

# Instale as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Exponha a porta em que o Flask irá rodar
EXPOSE 8000

# Comando para iniciar a aplicação Flask
CMD ["python", "main.py"]
