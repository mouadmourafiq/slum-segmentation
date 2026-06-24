import os
import shutil
import json
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import torch

import config
from inference import load_model, run_inference_smooth
from vectorize import vectorize_mask

app = FastAPI(title="Slum Segmentation API", description="AI powered slum detection")

# Global variables to hold the loaded model
ML_MODEL = None
DEVICE = None

@app.on_event("startup")
def load_ml_model():
    global ML_MODEL, DEVICE
    DEVICE = config.DEVICE
    model_path = config.MODEL_SAVE_PATH.replace(".pth", "_adapted.pth")
    if not os.path.exists(model_path):
        model_path = config.MODEL_SAVE_PATH
        
    print(f"[*] Starting API Server. Loading model from {model_path}...")
    ML_MODEL = load_model(model_path, DEVICE)
    print("[*] Model loaded successfully.")

# Mount the frontend directory to serve static files (CSS, JS)
frontend_dir = os.path.join(config.PROJECT_ROOT, "frontend")
os.makedirs(frontend_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Frontend not found. Please create index.html</h1>"

@app.post("/predict")
async def predict_slums(file: UploadFile = File(...)):
    print(f"[*] Received file for prediction: {file.filename}")
    
    # 1. Save uploaded file to disk
    input_path = os.path.join(config.DATA_DIR, "uploaded_image.tif")
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # 2. Define output paths
    output_mask = os.path.join(config.OUTPUT_DIR, "api_prediction.tif")
    output_geojson = os.path.join(config.OUTPUT_DIR, "api_prediction.geojson")
    
    # 3. Run Inference
    print("[*] Running AI inference...")
    run_inference_smooth(
        model=ML_MODEL,
        input_path=input_path,
        output_path=output_mask,
        device=DEVICE,
        tile_size=config.INFERENCE_TILE_SIZE,
        overlap=0.5,
        threshold=config.CONFIDENCE_THRESHOLD
    )
    
    # 4. Vectorize
    print("[*] Vectorizing prediction to GeoJSON...")
    vectorize_mask(output_mask, output_geojson)
    
    # 5. Return GeoJSON data to frontend
    if not os.path.exists(output_geojson):
        return JSONResponse(status_code=500, content={"error": "Vectorization failed. No GeoJSON produced."})
        
    with open(output_geojson, "r") as f:
        geojson_data = json.load(f)
        
    # Also return a URL to access the uploaded image if needed for display
    # We can just return the raw geojson.
    return {"status": "success", "geojson": geojson_data}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
