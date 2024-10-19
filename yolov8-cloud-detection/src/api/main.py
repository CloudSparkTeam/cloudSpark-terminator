from flask import Flask, request, jsonify, send_file
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import io

# Definindo o modelo UNet
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.decoder(x1)
        return x2

# Inicializando a aplicação Flask
app = Flask(__name__)

# Carregando o modelo
model = UNet(in_channels=3, out_channels=1)
model.load_state_dict(torch.load('unet_model.pth'))
model.eval()

# Definindo transformações para a entrada
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'files' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['files']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Processar a imagem
    input_image = Image.open(file).convert('RGB')
    original_size = input_image.size
    image = transform(input_image).unsqueeze(0)  # Adicionar uma dimensão de batch

    with torch.no_grad():
        output = model(image)

    # Converter a saída para uma imagem
    output_image = output.squeeze().numpy()  # Remover a dimensão do batch
    output_image = (output_image * 255).astype('uint8')  # Converter para 8 bits

    # Redimensionar a saída para o tamanho original da entrada
    mask_image = Image.fromarray(output_image).resize(original_size, Image.LANCZOS).convert('RGBA')

    # Converter a imagem original para RGBA
    input_image = input_image.convert('RGBA')

    # Mesclar as duas imagens com um nível de transparência (0.5 é 50%)
    blended = Image.blend(input_image, mask_image, alpha=0.5)

    # Salvar a imagem resultante em um buffer em memória
    img_io = io.BytesIO()
    blended.save(img_io, 'PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png', as_attachment=False, download_name='blended_output.png')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

