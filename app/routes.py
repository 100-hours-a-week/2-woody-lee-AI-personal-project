import os
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from PIL import Image
import io
from .model import load_model, get_transform
from .config import CLASS_NAMES

router = APIRouter()

model = load_model()
trnasform = get_transform()
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)



@router.post('/upload_image')
async def upload_image(file: UploadFile = File(...)):
    try:
        # 파일 내용 읽기
        contents = await file.read()
        # 저장할 파일 경로 설정
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        # 파일 저장
        with open(file_path, "wb") as f:
            f.write(contents)
        return {"message": "파일 업로드 성공", "filename": file.filename}
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": "파일 업로드 실패", "error": str(e)})



@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        
        image = Image.open(io.BytesIO(contents).convert("RGB"))
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "올바르지 않은 이미지 파일입니다.", "error": str(e)})
    
    input_tensor = get_transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        
    predicted_class = CLASS_NAMES[predicted.item()]
    return {"prediction": predicted_class}