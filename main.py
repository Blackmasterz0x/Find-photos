import io
import os
import glob
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from scipy.spatial.distance import cosine
import insightface
from insightface.app import FaceAnalysis

# ตั้งค่าแอปพลิเคชัน
app = FastAPI(title="STSS STUDIO - Face Search")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATASET_DIR = "dataset_images"

# อนุญาตให้หน้าเว็บโหลดรูปจากโฟลเดอร์ dataset_images ไปแสดงได้
if os.path.exists(DATASET_DIR):
    app.mount("/dataset_images", StaticFiles(directory=DATASET_DIR), name="dataset_images")

face_app = None
face_database = []

@app.on_event("startup")
def startup_event():
    global face_app, face_database
    print("🚀 กำลังเริ่มต้นระบบ AI และโหลดโมเดล InsightFace...")
    
    try:
        # โหลดโมเดล buffalo_l ด้วย CPU
        face_app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
        face_app.prepare(ctx_id=0, det_size=(640, 640))
        print("✅ โหลดโมเดล AI สำเร็จ")
    except Exception as e:
        print(f"❌ โหลดโมเดลไม่สำเร็จ: {e}")
        return

    # เช็คว่ามีโฟลเดอร์เก็บรูปไหม ถ้าไม่มีให้สร้างใหม่
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR, exist_ok=True)
        print(f"⚠️ ไม่พบโฟลเดอร์ {DATASET_DIR} ระบบสร้างให้แล้ว กรุณาใส่รูปภาพของคุณลงไป")
        return

    print(f"🔍 กำลังดึงข้อมูลรูปภาพจากโฟลเดอร์ '{DATASET_DIR}'...")
    valid_extensions = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.JPG", "*.PNG")
    image_paths = []
    for ext in valid_extensions:
        image_paths.extend(glob.glob(os.path.join(DATASET_DIR, ext)))

    print(f"พบไฟล์รูปภาพทั้งหมด {len(image_paths)} ไฟล์ กำลังเริ่มสกัดใบหน้าเพื่อสร้างฐานข้อมูล...")
    
    # วนลูปอ่านทุกรูปเพื่อแปลงเป็น Vector
    for img_path in image_paths:
        try:
            filename = os.path.basename(img_path)
            img = cv2.imread(img_path)
            
            # *** จุดที่แก้ BUG จากโค้ดเดิม ***
            if img is None:
                print(f"  [-] อ่านไฟล์ไม่ได้ (ไฟล์อาจเสียหาย): {filename}")
                continue
            
            faces = face_app.get(img)
            if len(faces) > 0:
                face_database.append({
                    "image_url": f"/dataset_images/{filename}",
                    "embedding": faces[0].embedding
                })
                print(f"  [+] สแกนสำเร็จ: {filename}")
            else:
                print(f"  [-] ไม่พบใบหน้าในรูป: {filename}")
        except Exception as e:
            print(f"  [-] ข้ามไฟล์ {img_path} (เกิดข้อผิดพลาด: {e})")

    print(f"🎉 ระบบพร้อมใช้งาน! นำเข้าฐานข้อมูลสำเร็จทั้งหมด {len(face_database)} รูป")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return HTMLResponse("<h1>Error: ไม่พบไฟล์ index.html ในระบบ</h1>", status_code=404)

@app.post("/search")
async def search_face(
    file: UploadFile = File(...),
    threshold: float = Form(0.55) # รับค่าความเหมือนจาก UI หน้าเว็บ
):
    # ตรวจสอบว่าเป็นไฟล์รูปไหม
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="กรุณาอัปโหลดไฟล์รูปภาพเท่านั้น (JPG, PNG, WEBP)")

    try:
        # อ่านไฟล์รูปที่ส่งมาจากหน้าเว็บ
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(image)
        
        # แปลง RGB กลับเป็น BGR เพื่อให้ InsightFace สแกนได้แม่นยำขึ้น
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # สแกนใบหน้า
        faces = face_app.get(img_bgr)
        if len(faces) == 0:
            return {
                "success": False, 
                "message": "AI ไม่พบใบหน้าในรูปภาพที่คุณอัปโหลด กรุณาใช้รูปหน้าตรง สว่าง และชัดเจน"
            }

        user_embedding = faces[0].embedding
        matches = []

        # เปรียบเทียบกับฐานข้อมูล
        for item in face_database:
            dist = cosine(user_embedding, item["embedding"])
            similarity = max(0.0, float(1.0 - dist))
            
            # ถ้าระยะห่าง (dist) น้อยกว่าระดับความเข้มงวด (threshold) ที่กำหนด
            if dist < threshold:
                matches.append({
                    "image_url": item["image_url"],
                    "similarity": round(similarity * 100, 1)
                })

        # เรียงลำดับจากรูปที่เหมือนที่สุดไปน้อยที่สุด
        matches.sort(key=lambda x: x["similarity"], reverse=True)

        return {
            "success": True,
            "total_matched": len(matches),
            "results": matches
        }

    except Exception as e:
        print(f"API Search Error: {e}")
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาดทางเซิร์ฟเวอร์: {str(e)}")