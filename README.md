
# API de Segmentação de Nuvens

Esta API fornece uma funcionalidade para segmentação de nuvens em imagens utilizando um modelo treinado U-Net. Ela recebe imagens, realiza a segmentação e retorna os resultados como arquivos GeoTIFF ou PNG.

## Endpoints

### 1. `/predict` - Segmentação de Imagens

Realiza a segmentação de uma ou mais imagens enviadas, retornando os resultados como arquivos GeoTIFF compactados em um arquivo ZIP.

#### Método HTTP
`POST`

#### Parâmetros de Requisição

- **files**: Um ou mais arquivos de imagem. As imagens devem ter 4 canais (RGB + NIR).
  
  **Tipo**: Arquivo (pode enviar múltiplos arquivos)

#### Exemplo de Requisição

```bash
curl -X POST "http://127.0.0.1:8000/predict" -F "files=@/caminho/para/imagem1.png" -F "files=@/caminho/para/imagem2.png"
```

#### Resposta

- **200 OK**: Retorna um arquivo ZIP contendo os arquivos GeoTIFF gerados.
- **400 Bad Request**: Caso não haja arquivos na requisição.
- **500 Internal Server Error**: Caso ocorra um erro ao processar a imagem.

**Exemplo de Resposta (200 OK)**

O arquivo ZIP será retornado com o nome `segmented_images.zip`, contendo os arquivos GeoTIFF gerados pela segmentação.

---

### 2. `/convert_to_images` - Converter GeoTIFF para PNG

Este endpoint converte arquivos GeoTIFF enviados em um arquivo ZIP para imagens PNG. O arquivo ZIP de saída será retornado.

#### Método HTTP
`POST`

#### Parâmetros de Requisição

- **file**: Um arquivo ZIP contendo arquivos GeoTIFF.
  
  **Tipo**: Arquivo (um arquivo ZIP contendo GeoTIFFs)

#### Exemplo de Requisição

```bash
curl -X POST "http://127.0.0.1:8000/convert_to_images" -F "file=@/caminho/para/arquivo.zip"
```

#### Resposta

- **200 OK**: Retorna um arquivo ZIP contendo os arquivos PNG convertidos.
- **400 Bad Request**: Caso não haja arquivos na requisição ou o arquivo enviado não seja um ZIP.
- **500 Internal Server Error**: Caso ocorra um erro ao processar o arquivo.

**Exemplo de Resposta (200 OK)**

O arquivo ZIP será retornado com o nome `converted_images.zip`, contendo as imagens PNG convertidas a partir dos arquivos GeoTIFF.

---

## Detalhes de Implementação

### Dependências

Esta API foi construída com os seguintes pacotes:

- **Torch:** Para inferência do modelo U-Net.
- **Flask:** Framework para criação da API.
- **GeoPandas:** Para manipulação e exportação de arquivos GeoTIFF.
- **Shapely:** Para operações geométricas.

### Como rodar a API localmente

1. Clone o repositório e crie um ambiente virtual:
    ```bash
    git clone <URL do repositório>
    cd <diretório do projeto>
    python -m venv venv
    ```

2. Ative o ambiente virtual:
    - No Windows:
      ```bash
      .\venv\Scripts\Activate
      ```
    - No Linux/macOS:
      ```bash
      source venv/bin/activate
      ```

3. Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```

4. Execute a API:
    ```bash
    python main.py
    ```

A API será iniciada no `http://127.0.0.1:8000`.

### Testando a API

Você pode testar a API usando o `curl`, Postman, ou qualquer outro cliente HTTP para enviar requisições `POST` com arquivos de imagem ou ZIP, conforme descrito acima.

### Modelos

A API usa um modelo de segmentação de nuvens baseado na arquitetura U-Net, treinado com dados de imagens contendo 4 canais: RGB e Near Infrared (NIR). O modelo é carregado e usado para fazer previsões de segmentação nas imagens enviadas.

---

## Licença

Esta API é fornecida sob a [Licença MIT](LICENSE), o que permite o uso, cópia, modificação e distribuição do software de acordo com as condições especificadas.
