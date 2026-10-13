from pathlib import Path
import base64, io, os, sys, tempfile, traceback
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR / "src"))
import predict_proposed as predictor

app = FastAPI(title="PEP-CV Pet Screening API", version="1.0.0")
origins = [x.strip() for x in os.getenv("FRONTEND_ORIGIN", "*").split(",") if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins or ["*"],
                   allow_credentials=False, allow_methods=["GET", "POST", "OPTIONS"],
                   allow_headers=["*"])

@app.get("/")
def root():
    return {"service": "PEP-CV API", "status": "ok"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")
    data = await file.read()
    if len(data) > 12 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image must be 12 MB or smaller.")
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")
    output_dir = Path(tempfile.mkdtemp(prefix="pep_cv_"))
    try:
        modality, routing_confidence = predictor.predict_modality(image)
        if modality == "eye":
            result = predictor.run_eye_mcfa(image, output_dir)
        elif modality == "skin":
            result = predictor.run_skin_cstf(image, output_dir)
        else:
            raise RuntimeError(f"Unexpected modality returned: {modality}")
        cam_path = Path(result["layercam_path"])
        if not cam_path.is_file():
            raise RuntimeError("LayerCAM image was not generated.")
        cam_data = "data:image/png;base64," + base64.b64encode(cam_path.read_bytes()).decode("ascii")
        top_k = [{"class": str(x["class"]), "confidence": float(x["confidence"])}
                 for x in result.get("top_k", [])]
        return {
            "modality": modality,
            "routing_confidence": float(routing_confidence),
            "model": str(result.get("model", "")),
            "prediction": {"class": str(result["prediction"]["class"]),
                           "confidence": float(result["prediction"]["confidence"])},
            "top_k": top_k,
            "layercam": cam_data,
            "notice": "This is an AI screening aid, not a veterinary diagnosis."
        }
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
