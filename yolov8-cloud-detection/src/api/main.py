from flask import Flask, request, jsonify, send_file
import pyproj
import numpy as np
import tifffile as tiff  # Para ler arquivos TIFF
import torch  # Para carregar o modelo PyTorch
import os
import tempfile

app = Flask(__name__)

# Carrega o modelo
model = torch.load("best_model.pth")
model.eval()  # Define o modelo para o modo de avaliação

@app.route('/predict', methods=['POST'])
def predict():
    # Verifica se o arquivo foi enviado
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhum arquivo fornecido.'}), 400

    file = request.files['file']

    # Verifica se o arquivo é um TIFF
    if not file.filename.endswith('.tiff') and not file.filename.endswith('.tif'):
        return jsonify({'error': 'Arquivo não é um TIFF.'}), 400

    # Salva o arquivo temporariamente
    temp_path = tempfile.mktemp(suffix='.tif')
    file.save(temp_path)

    try:
        # Carrega a imagem TIFF
        with tiff.TiffFile(temp_path) as tif_file:
            image_data = tif_file.asarray()  # Obtém os dados da imagem
            
            # Processa a imagem usando o modelo
            input_tensor = preprocess_image(image_data)  # Função de pré-processamento que você precisa definir
            with torch.no_grad():
                output_tensor = model(input_tensor)  # Faz a previsão

            # Converte a saída para um formato adequado (por exemplo, numpy array)
            output_array = output_tensor.numpy()  # Ou outro método para converter conforme necessário
            
            # Salva o resultado como GeoTIFF
            output_path = tempfile.mktemp(suffix='.tif')
            tiff.imsave(output_path, output_array)

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Remove o arquivo temporário
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return send_file(output_path, mimetype='image/tiff'), 200  # Envia o GeoTIFF como resposta

def preprocess_image(image_data):
    # Implemente aqui a função para pré-processar a imagem conforme necessário para o seu modelo
    # Isso pode incluir redimensionamento, normalização, etc.
    # Exemplo (ajuste conforme necessário):
    image_data = image_data / 255.0  # Normalização
    image_data = np.transpose(image_data, (2, 0, 1))  # Transpor para C x H x W
    input_tensor = torch.tensor(image_data, dtype=torch.float32).unsqueeze(0)  # Adiciona a dimensão do batch
    return input_tensor

if __name__ == '__main__':
    # Configurando para rodar em qualquer IP na porta 8000
    app.run(host='0.0.0.0', port=8000)
