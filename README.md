# percepta_patentYL

Worker RunPod Serverless para deteccion de patentes con YOLO11.  
Forma parte del proyecto **Percepta** (UTN — Introduccion a la Vision Artificial).

## Que hace

Recibe un frame de camara codificado en base64, ejecuta inferencia con el modelo `morsetechlab/yolov11-license-plate-detection` en GPU, y devuelve las regiones de patentes detectadas en formato JSON con score y bounding box.

## Endpoint

```
POST https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync
Authorization: Bearer {API_TOKEN_RUNPOD}
Content-Type: application/json
```

### Request

```json
{
  "input": {
    "image_base64": "<base64 del frame JPEG>",
    "conf": 0.35,
    "imgsz": 640
  }
}
```

Alternativa: reemplazar `image_base64` por `image_url` con una URL publica.

### Response

```json
{
  "output": {
    "detections": [
      {
        "label": "license-plate",
        "score": 0.91,
        "bbox_xyxy": [142, 310, 398, 374]
      }
    ],
    "n_detecciones": 1,
    "modelo": "morsetechlab/yolov11-license-plate-detection"
  }
}
```

## Stack

| Componente | Version |
|---|---|
| Base image | `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` |
| Python | 3.11 (provisto por la base image) |
| torch | 2.5.1+cu124 (provisto por la base image, no reinstalado) |
| ultralytics | 8.3.31 |
| numpy | 1.26.4 |
| runpod | 1.7.13 |
| Modelo | morsetechlab/yolov11-license-plate-detection |
| GPU worker | 16 GB (RunPod Serverless) |

## Archivos

| Archivo | Descripcion |
|---|---|
| `handler.py` | Entry point del worker RunPod; carga YOLO11 y expone `handler()` |
| `Dockerfile` | Build de la imagen; incluye libs de sistema, checks fail-fast y precarga el modelo |
| `requirements.txt` | Dependencias pip (sin torch — lo provee la base image) |
| `.github/workflows/deploy.yml` | CI/CD: build → GHCR → RunPod en cada push a main |

## Build y deploy

El build se dispara automaticamente al hacer push a `main` via GitHub Actions:

1. Construye la imagen Docker
2. La pushea a GHCR (`ghcr.io/portfoliofullstack/percepta_patentyl`)
3. Crea/actualiza el template y endpoint en RunPod via GraphQL API

## Variables de entorno del worker

| Variable | Descripcion | Default |
|---|---|---|
| `CONF` | Umbral de confianza minimo | `0.35` |
| `YOLO_AUTOINSTALL` | Desactiva auto-update de ultralytics en runtime | `False` |

## Secrets requeridos en GitHub Actions

| Secret | Descripcion |
|---|---|
| `RUNPOD_API_KEY` | Token de acceso a la API de RunPod |
| `HF_TOKEN` | Token de Hugging Face (opcional, modelo es publico) |

## Notas de implementacion

- `torch` y `torchvision` **no** se incluyen en `requirements.txt`. La base image `pytorch/pytorch` los provee via conda; reinstalarlos con pip genera conflictos (`RuntimeError: Numpy is not available`).
- `YOLO_AUTOINSTALL=False` es obligatorio. Sin esta variable, ultralytics instala dependencias al importar y corrompe el entorno pip.
- La lib de sistema correcta para Ubuntu 22.04 es `libtiff5`, no `libtiff6`.
- El modelo se precarga en tiempo de build (`snapshot_download`) para eliminar cold start.
- Se incluye un check `assert` post-descarga para fallar en build time si el `.pt` no quedo en la imagen.

## Cliente de referencia

El dashboard que consume este worker esta en:  
`tpFinal_imp/client/dashboard.py`
