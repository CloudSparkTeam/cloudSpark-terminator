import tifffile as tiff
from fastapi import FastAPI, File, UploadFile, HTTPException
from ultralytics import YOLO
from fastapi.responses import FileResponse
from PIL import Image, ImageDraw
import io
import os
import uuid
import numpy as np
from typing import List

app = FastAPI()

# Carregar o modelo treinado
model = YOLO('runs/train/cloud-detection/weights/best.pt')

# Pasta temporária para salvar imagens processadas
TEMP_DIR = "temp_images"
os.makedirs(TEMP_DIR, exist_ok=True)

# Tamanho da imagem que foi utilizado no treinamento
TARGET_IMG_SIZE = 640  # 640x640

@app.post("/predict/")
async def predict_images(files: List[UploadFile] = File(...)):
    processed_images = []

    for file in files:
        # Se for um arquivo .tif
        if file.filename.endswith(".tif"):
            # Ler a imagem TIFF usando tifffile
            tiff_image = tiff.imread(io.BytesIO(await file.read()))

            # Exibir as propriedades da imagem para diagnóstico
            print(f"Dimensões da Imagem: {tiff_image.shape}")
            print(f"Tipo de dados: {tiff_image.dtype}")

            # Se a imagem tiver múltiplas bandas, converter para RGB
            if tiff_image.ndim == 3:
                image_rgb = np.stack([tiff_image[:, :, 0], tiff_image[:, :, 1], tiff_image[:, :, 2]], axis=-1)
            elif tiff_image.ndim == 2:
                # Se for uma imagem de banda única (exemplo: BAND5), replicar para formar RGB
                image_rgb = np.stack([tiff_image, tiff_image, tiff_image], axis=-1)  # Replicar a banda para RGB
            else:
                raise HTTPException(status_code=400, detail="Formato da imagem não suportado ou imagem corrompida.")

            # Converter para imagem PIL
            image_pil = Image.fromarray(image_rgb)

        # Se for um arquivo .png
        else:
            # Ler a imagem PNG usando PIL
            image_pil = Image.open(io.BytesIO(await file.read()))
            # Se a imagem não for RGB, converter
            if image_pil.mode != 'RGB':
                image_pil = image_pil.convert('RGB')

        # Redimensionar a imagem para o tamanho utilizado no treinamento (640x640)
        image_pil = image_pil.resize((TARGET_IMG_SIZE, TARGET_IMG_SIZE))

        # Fazer a inferência usando o YOLOv8
        results = model.predict(image_pil, conf=0.5, iou=0.4)
        
        # Marcar as caixas de detecção diretamente na imagem
        draw = ImageDraw.Draw(image_pil)
        for box in results[0].boxes.xyxy.tolist():
            draw.rectangle(box, outline="red", width=3)

        # Gerar um nome de arquivo único para salvar a imagem processada
        output_filename = f"{uuid.uuid4()}.png"
        output_path = os.path.join(TEMP_DIR, output_filename)

        # Salvar a imagem com as detecções
        image_pil.save(output_path)
        processed_images.append(output_path)

    # Retornar as imagens processadas
    if len(processed_images) == 1:
        # Retorna uma única imagem
        return FileResponse(processed_images[0], media_type="image/png", filename=processed_images[0])
    else:
        # Retorna múltiplas imagens
        return [FileResponse(image_path, media_type="image/png", filename=os.path.basename(image_path)) for image_path in processed_images]
