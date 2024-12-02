import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from flask import Flask, request, jsonify, send_file
from io import BytesIO
import tempfile
from torchvision.models import resnet34

# Aumenta o limite de pixels
Image.MAX_IMAGE_PIXELS = None

# Verifica se o modelo já está treinado e se o GPU está disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Função para detectar áreas estouradas (branco saturado)
def detect_overexposed_areas(image_array, threshold=240):
    # Identifica os pixels acima do limiar (por exemplo, 240 para áreas muito brancas)
    overexposed_mask = image_array > threshold
    return overexposed_mask.astype(np.uint8) * 255  # Cria uma máscara binária

# Função para sobrepor a máscara na imagem original
def overlay_mask_on_image(original_path, mask_array, alpha=0.5):
    original_image = Image.open(original_path).convert("RGBA")
    mask_image = Image.fromarray(mask_array).convert("L")  
    mask_image = ImageOps.colorize(mask_image, black="black", white="white")  
    mask_image = mask_image.convert("RGBA")
    mask_image = mask_image.resize(original_image.size)
    mask_image.putalpha(int(alpha * 255))

    combined_image = Image.alpha_composite(original_image, mask_image)
    return combined_image

# Função para recortar tensores
def crop_tensor(tensor, target_tensor):
    target_height, target_width = target_tensor.shape[2], target_tensor.shape[3]
    tensor_height, tensor_width = tensor.shape[2], tensor.shape[3]
    delta_height = tensor_height - target_height
    delta_width = tensor_width - target_width

    crop_top = delta_height // 2
    crop_bottom = delta_height - crop_top
    crop_left = delta_width // 2
    crop_right = delta_width - crop_left

    return tensor[:, :, crop_top:tensor_height - crop_bottom, crop_left:tensor_width - crop_right]

# Função para carregar e aplicar o contraste no TIFF
def load_and_apply_contrast(image_path, contrast_factor=2.0):
    image = Image.open(image_path)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast_factor)
    return np.array(image)

# Arquitetura do U-Net
class UNET(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        base_model = resnet34(weights=None)
        base_model.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.base_layers = list(base_model.children())
        
        self.layer0 = nn.Sequential(*self.base_layers[:3])
        self.layer1 = nn.Sequential(*self.base_layers[3:5])
        self.layer2 = self.base_layers[5]
        self.layer3 = self.base_layers[6]
        self.layer4 = self.base_layers[7]

        self.upconv4 = self.expand_block(512, 256)
        self.upconv3 = self.expand_block(256 * 2, 128)
        self.upconv2 = self.expand_block(128 * 2, 64)
        self.upconv1 = self.expand_block(64 * 2, 64)
        self.upconv0 = self.expand_block(64 * 2, out_channels)

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        upconv4 = self.upconv4(layer4)
        upconv3 = self.upconv3(torch.cat([crop_tensor(upconv4, layer3), layer3], 1))
        upconv2 = self.upconv2(torch.cat([crop_tensor(upconv3, layer2), layer2], 1))
        upconv1 = self.upconv1(torch.cat([crop_tensor(upconv2, layer1), layer1], 1))
        upconv0 = self.upconv0(torch.cat([crop_tensor(upconv1, layer0), layer0], 1))

        return upconv0

    def expand_block(self, in_channels, out_channels):
        expand = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        return expand

# Inicializa a aplicação Flask
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 * 1024

# Carrega o modelo treinado
model = UNET(in_channels=4, out_channels=2)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

# Função para transformar predição em máscara binária
def predb_to_mask(predb, idx):
    p = F.softmax(predb[idx], dim=0)
    mask = p.argmax(0).cpu().numpy()
    return (mask == 1).astype(np.uint8) * 255

# Atualizando a função segment_image para a API
def segment_image(image_path):
    image_array = load_and_apply_contrast(image_path, contrast_factor=2.0)
    image_tensor = torch.tensor(image_array).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

    with torch.no_grad():
        predb = model(image_tensor)

    output_image = predb_to_mask(predb, 0)
    return output_image

@app.route('/tiff/segmented', methods=['POST'])
def generate_segmented_tiff():
    if 'files' not in request.files:
        return jsonify({'error': 'No files part'}), 400

    files = request.files['files']
    if files.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Salva o arquivo temporariamente
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        files.save(temp_file.name)
        segmented_mask = segment_image(temp_file.name)

    # Salva a máscara segmentada como TIFF
    segmented_output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".tiff").name
    Image.fromarray(segmented_mask).save(segmented_output_path, format='TIFF')

    # Retorna o TIFF gerado
    return send_file(segmented_output_path, mimetype='image/tiff', as_attachment=True, download_name='segmented_mask.tiff')

@app.route('/tiff/overexposed', methods=['POST'])
def generate_overexposed_tiff():
    if 'files' not in request.files:
        return jsonify({'error': 'No files part'}), 400

    files = request.files['files']
    if files.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Salva o arquivo temporariamente
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        files.save(temp_file.name)

        # Carregar a imagem original e processar a máscara
        original_image = Image.open(temp_file.name)
        overexposed_mask = detect_overexposed_areas(np.array(original_image), threshold=240)

    # Salva a máscara de áreas estouradas como TIFF
    overexposed_output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".tiff").name
    Image.fromarray(overexposed_mask).save(overexposed_output_path, format='TIFF')

    # Retorna o TIFF gerado
    return send_file(overexposed_output_path, mimetype='image/tiff', as_attachment=True, download_name='overexposed_mask.tiff')

@app.route('/thumbnail/segmented', methods=['POST'])
def generate_segmented_thumbnail():
    if 'files' not in request.files:
        return jsonify({'error': 'No files part'}), 400

    files = request.files['files']
    if files.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Salva o arquivo temporariamente
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        files.save(temp_file.name)
        segmented_mask = segment_image(temp_file.name)

    # Salva o thumbnail da máscara segmentada como PNG
    segmented_thumbnail = BytesIO()
    Image.fromarray(segmented_mask).save(segmented_thumbnail, format='PNG')
    segmented_thumbnail.seek(0)

    # Retorna o thumbnail gerado
    return send_file(segmented_thumbnail, mimetype='image/png', as_attachment=True, download_name='segmented_thumbnail.png')


@app.route('/thumbnail/overexposed', methods=['POST'])
def generate_overexposed_thumbnail():
    if 'files' not in request.files:
        return jsonify({'error': 'No files part'}), 400

    files = request.files['files']
    if files.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Salva o arquivo temporariamente
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        files.save(temp_file.name)

        # Carregar a imagem original e processar a máscara
        original_image = Image.open(temp_file.name)
        overexposed_mask = detect_overexposed_areas(np.array(original_image), threshold=240)

    # Salva o thumbnail da máscara de áreas estouradas como PNG
    overexposed_thumbnail = BytesIO()
    Image.fromarray(overexposed_mask).save(overexposed_thumbnail, format='PNG')
    overexposed_thumbnail.seek(0)

    # Retorna o thumbnail gerado
    return send_file(overexposed_thumbnail, mimetype='image/png', as_attachment=True, download_name='overexposed_thumbnail.png')



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
