"""
RunPod Serverless Worker — Detección de patentes con YOLO11.
Modelo: morsetechlab/yolov11-license-plate-detection

Input:
  {
    "input": {
      "image_base64": "<base64 JPEG>",
      "conf":  0.35,
      "imgsz": 640
    }
  }

Output:
  {
    "detections": [
      { "bbox_xyxy": [x1, y1, x2, y2], "score": 0.91, "label": "license-plate" }
    ],
    "n_detecciones": 1,
    "modelo": "morsetechlab/yolov11-license-plate-detection"
  }
"""
import base64
import glob
import io
import os

import numpy as np
import runpod
from PIL import Image
from ultralytics import YOLO

MODEL_REPO  = "morsetechlab/yolov11-license-plate-detection"
MODEL_DIR   = "/app/models/license-plate"
DEFAULT_CONF = float(os.getenv("CONF", "0.35"))

# ── Carga al arrancar el worker (una sola vez, a nivel de módulo) ─────────────
_weights = glob.glob(os.path.join(MODEL_DIR, "**/*.pt"), recursive=True)
if not _weights:
    raise FileNotFoundError(f"No se encontró .pt en {MODEL_DIR}")

model = YOLO(_weights[0])
print(f"[WORKER] Modelo cargado: {_weights[0]}")


# ── Handler ───────────────────────────────────────────────────────────────────

def handler(event: dict) -> dict:
    inp   = event.get("input", {}) if isinstance(event, dict) else {}
    conf  = float(inp.get("conf",  DEFAULT_CONF))
    imgsz = int(inp.get("imgsz",  640))

    if inp.get("image_base64"):
        b64 = inp["image_base64"]
        if "," in b64 and b64.strip().lower().startswith("data:"):
            b64 = b64.split(",", 1)[1]
        try:
            img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        except Exception as e:
            return {"error": f"Error decodificando image_base64: {e}"}
    elif inp.get("image_url"):
        import requests
        try:
            r = requests.get(inp["image_url"], timeout=30)
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
        except Exception as e:
            return {"error": f"Error descargando image_url: {e}"}
    else:
        return {"error": "Requerido: 'image_base64' o 'image_url' en input"}

    # fuerza numpy antes de inferencia — falla rápido si algo está roto
    _ = np.asarray(img)

    try:
        results = model.predict(img, imgsz=imgsz, conf=conf, verbose=False)
        r0      = results[0]
    except Exception as e:
        return {"error": f"Error en inferencia: {e}"}

    detections = []
    if r0.boxes is not None and len(r0.boxes) > 0:
        for b in r0.boxes:
            cls_id = int(b.cls[0].item())
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            detections.append({
                "label":     r0.names.get(cls_id, str(cls_id)),
                "score":     round(float(b.conf[0].item()), 4),
                "bbox_xyxy": [round(x1), round(y1), round(x2), round(y2)],
            })

    return {
        "detections":    detections,
        "n_detecciones": len(detections),
        "modelo":        MODEL_REPO,
    }


runpod.serverless.start({"handler": handler})
