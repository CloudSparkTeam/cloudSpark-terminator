import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_file
from io import BytesIO
import geopandas as gpd
from shapely.geometry import box
import tempfile
import zipfile
from torchvision.models import resnet34, ResNet34_Weights

# Verifica se o modelo já está treinado e se o GPU está disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Arquitetura do U-Net conforme seu código
class UNET(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Modifica a primeira camada do ResNet34 para aceitar um número customizado de canais
        base_model = resnet34(weights=None)  # Não carregar o modelo pré-treinado
        base_model.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.base_layers = list(base_model.children())
        
        self.layer0 = nn.Sequential(*self.base_layers[:3]) 
        self.layer1 = nn.Sequential(*self.base_layers[3:5])
        self.layer2 = self.base_layers[5]  
        self.layer3 = self.base_layers[6]
        self.layer4 = self.base_layers[7]
        
        self.upconv4 = self.expand_block(512, 256)
        self.upconv3 = self.expand_block(256*2, 128)
        self.upconv2 = self.expand_block(128*2, 64)
        self.upconv1 = self.expand_block(64*2, 64)
        self.upconv0 = self.expand_block(64*2, out_channels)
    
    def forward(self, x):
        # Caminho contração
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        
        # Caminho expansivo
        upconv4 = self.upconv4(layer4)
        upconv3 = self.upconv3(torch.cat([upconv4, layer3], 1))
        upconv2 = self.upconv2(torch.cat([upconv3, layer2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, layer1], 1))
        upconv0 = self.upconv0(torch.cat([upconv1, layer0], 1))
    
        return upconv0
    
    def expand_block(self, in_channels, out_channels):
        expand = nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
        )
        return expand

# Inicializa a aplicação Flask
app = Flask(__name__)

# Carrega o modelo treinado
model = UNET(in_channels=4, out_channels=2)  # Modelo com 4 canais de entrada (incluindo NIR)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()  # Coloca o modelo em modo de avaliação

# Função para fazer a segmentação em uma imagem
def segment_image(image_path):
    # Carrega e prepara a imagem
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    
    # Normaliza a imagem para os valores que o modelo espera
    image = torch.tensor(image).unsqueeze(0).float().to(device) / 255.0
    if image.shape[1] != 4:  # Verifica se a imagem tem 4 canais (RGB + NIR)
        raise ValueError("A imagem precisa ter 4 canais (RGB + NIR).")
    
    # Faz a previsão
    with torch.no_grad():
        output = model(image)
    
    # Converte a saída para uma imagem
    output = output.squeeze().cpu().numpy()  # Remove batch dimension
    output_image = np.argmax(output, axis=0)  # A saída pode ser uma máscara com probabilidades
    
    return output_image

# Função para salvar a segmentação como GeoTIFF usando geopandas
def save_as_geotiff(mask, transform, crs='EPSG:4326'):
    """
    Salva a máscara de segmentação como um arquivo GeoTIFF e retorna o caminho temporário.
    Args:
    - mask: A máscara de segmentação (imagem 2D).
    - transform: Transformação geoespacial (metadados de espaçamento e posição).
    - crs: Sistema de referência de coordenadas (padrão é EPSG:4326).
    """
    # Converte a máscara em um GeoDataFrame
    width, height = mask.shape
    # Cria uma geometria para a máscara (usando bounding box para o raster)
    minx, miny = transform * (0, 0)
    maxx, maxy = transform * (width, height)
    geom = box(minx, miny, maxx, maxy)
    
    gdf = gpd.GeoDataFrame(
        {'geometry': [geom], 'mask': [mask]},
        crs=crs
    )
    
    # Cria um arquivo temporário para o GeoTIFF
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmpfile:
        geo_tiff_path = tmpfile.name
        
        # Salva o GeoDataFrame como GeoTIFF
        gdf.to_file(geo_tiff_path, driver='GTiff', index=False)
        
        return geo_tiff_path

# Função para converter GeoTIFF para imagem PNG
def geotiff_to_image(geotiff_path):
    with gpd.read_file(geotiff_path) as src:
        # Lê a máscara do GeoTIFF
        mask = src['mask'].values[0]
        image = Image.fromarray(mask.astype(np.uint8))  # Converte para imagem PIL
        return image

# Rota para segmentar várias imagens
@app.route('/predict', methods=['POST'])
def predict():
    if 'files' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    files = request.files.getlist('files')  # Pega a lista de arquivos
    
    if not files:
        return jsonify({'error': 'No selected files'}), 400
    
    result_files = []  # Lista para armazenar os arquivos gerados
    
    for file in files:
        if file.filename == '':
            continue  # Ignora arquivos sem nome
        
        # Salva o arquivo temporariamente
        temp_file_path = os.path.join(tempfile.mkdtemp(), file.filename)
        file.save(temp_file_path)
        
        # Faz a segmentação
        try:
            result = segment_image(temp_file_path)
            
            # Definir transformação geoespacial (exemplo)
            # Isso deve ser configurado conforme os dados reais (metadados da imagem original)
            transform = from_origin(west=0, north=100, xsize=30, ysize=30)  # Exemplo de transformação (modifique conforme necessário)
            
            # Salva a máscara como GeoTIFF temporário
            geo_tiff_path = save_as_geotiff(result, transform)
            result_files.append(geo_tiff_path)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    # Após o processamento, retorna os arquivos gerados como uma resposta zipada
    # Usando send_file para enviar todos os arquivos temporários zipados
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for geo_tiff in result_files:
            zip_file.write(geo_tiff, os.path.basename(geo_tiff))
            os.remove(geo_tiff)  # Exclui o arquivo temporário após adicionar ao zip
    
    zip_buffer.seek(0)
    
    return send_file(zip_buffer, mimetype='application/zip', as_attachment=True, download_name='segmented_images.zip')


# Nova rota para converter arquivos ZIP contendo GeoTIFF para imagens PNG
@app.route('/convert_to_images', methods=['POST'])
def convert_to_images():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Salva o arquivo ZIP temporariamente
    temp_zip_path = os.path.join(tempfile.mkdtemp(), file.filename)
    file.save(temp_zip_path)
    
    # Extraí os GeoTIFFs do arquivo ZIP
    extracted_files = []
    with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_zip_path)
        extracted_files = zip_ref.namelist()
    
    # Converte cada GeoTIFF extraído para PNG
    png_files = []
    for geo_tiff in extracted_files:
        geo_tiff_path = os.path.join(temp_zip_path, geo_tiff)
        try:
            image = geotiff_to_image(geo_tiff_path)
            png_path = geo_tiff_path.replace('.tif', '.png')
            image.save(png_path)
            png_files.append(png_path)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    # Retorna os arquivos PNG gerados em um arquivo zip
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for png_file in png_files:
            zip_file.write(png_file, os.path.basename(png_file))
            os.remove(png_file)  # Exclui o arquivo PNG temporário
    
    zip_buffer.seek(0)
    
    return send_file(zip_buffer, mimetype='application/zip', as_attachment=True, download_name='converted_images.zip')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
