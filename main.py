import io, os, time
from collections import defaultdict, deque

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from insightface.app import FaceAnalysis
from PIL import Image, ImageOps

MODEL = os.getenv("MODEL_NAME", "buffalo_sc")
CODE = os.getenv("ACCESS_CODE", "")                       # ตั้งค่า = ต้องกรอกรหัสก่อนค้นหา
ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")    # ใส่โดเมนโรงเรียนเมื่อแยกโฮสต์ UI
MAX_BYTES, RATE_PER_MIN = 10 * 1024 * 1024, 12

app = FastAPI(title="STSS STUDIO Face Search")
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["*"], allow_headers=["*"])
S = {"fa": None, "emb": np.zeros((0, 512), np.float32), "owner": np.zeros(0, np.int32), "names": np.array([], dtype=str), "scale": 1.0, "src": "local"}
EMB_URL = os.getenv("EMB_URL", "")   # ถ้าไฟล์ embeddings ใหญ่เกินใส่ใน GitHub: ใส่ลิงก์ดาวน์โหลดตรงที่นี่
hits = defaultdict(deque)


@app.on_event("startup")
def startup():
    fa = FaceAnalysis(name=MODEL, allowed_modules=["detection", "recognition"], providers=["CPUExecutionProvider"])
    fa.prepare(ctx_id=0, det_size=(640, 640))
    S["fa"] = fa
    path = "data/embeddings.npz"
    if not os.path.exists(path) and EMB_URL:
        import urllib.request
        os.makedirs("data", exist_ok=True); urllib.request.urlretrieve(EMB_URL, path)
    if not os.path.exists(path):
        print("⚠️ ไม่พบ data/embeddings.npz — รัน build_index.py ก่อน")
        return
    
    z = np.load(path)
    
    # --- ส่วนที่แก้ไขให้เข้ากับฐานข้อมูลจาก Colab ---
    emb_data = z["embeddings"] if "embeddings" in z else z["emb"]
    owner_data = z["owner"] if "owner" in z else np.arange(len(emb_data), dtype=np.int32)
    
    S.update(emb=emb_data, 
             owner=owner_data, 
             names=z["names"], 
             src=str(z["source"]) if "source" in z else "local",
             scale=255.0 if emb_data.dtype == np.int8 else 1.0)
             
    print(f"✅ พร้อมใช้งาน: {len(S['names'])} รูป / {len(S['emb'])} ใบหน้า / โมเดล {MODEL}")
    # ----------------------------------------------


def urls(n):
    if S["src"] == "drive":   # รูปอยู่ใน Google Drive: ส่งแค่รหัสไฟล์ ไม่ต้องโฮสต์รูปเอง
        return {"thumb": f"https://drive.google.com/thumbnail?id={n}&sz=w480", "full": f"https://drive.google.com/thumbnail?id={n}&sz=w1600",
                "open": f"https://drive.google.com/file/d/{n}/view", "download": f"https://drive.google.com/uc?export=download&id={n}"}
    return {"thumb": f"/thumbs/{n}", "full": f"/photos/{n}", "open": f"/photos/{n}", "download": f"/photos/{n}"}


def score(q):   # คำนวณเป็นก้อน ๆ ประหยัด RAM แม้มีหลายหมื่นใบหน้า
    out = np.empty(len(S["emb"]), np.float32)
    for a in range(0, len(out), 20000):
        out[a:a + 20000] = S["emb"][a:a + 20000].astype(np.float32) @ q / S["scale"]
    return out


def guard(request: Request, code: str):
    if CODE and code != CODE:
        raise HTTPException(401, "รหัสเข้าใช้งานไม่ถูกต้อง")
    ip = (request.headers.get("x-forwarded-for") or request.client.host).split(",")[0].strip()
    q, now = hits[ip], time.time()
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE_PER_MIN:
        raise HTTPException(429, "ค้นหาถี่เกินไป รอ 1 นาทีแล้วลองใหม่")
    q.append(now)


@app.get("/api/info")
def info():
    return {"ready": S["fa"] is not None, "photos": int(len(S["names"])), "needs_code": bool(CODE), "model": MODEL}


@app.post("/api/search")
def search(request: Request, file: UploadFile = File(...), min_similarity: float = Form(0.45), code: str = Form("")):
    guard(request, code)
    raw = file.file.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "ไฟล์ใหญ่เกิน 10MB")
    try:
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    except Exception:
        raise HTTPException(400, "อ่านไฟล์รูปไม่ได้ ลองใช้ JPG, PNG หรือ WEBP")
    if len(S["names"]) == 0:
        return {"success": False, "message": "ยังไม่มีรูปในคลัง กรุณาติดต่อผู้ดูแลระบบ"}
    im.thumbnail((1280, 1280))
    faces = S["fa"].get(cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR))
    if not faces:
        return {"success": False, "message": "ไม่พบใบหน้าในรูป ลองใช้รูปที่เห็นหน้าตรง สว่าง และชัดเจน"}

    me = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))   # ใบหน้าที่ใหญ่สุด
    sims = score(me.normed_embedding.astype(np.float32))                       # cosine similarity ทุกใบหน้าในคลัง
    best = np.full(len(S["names"]), -1.0, np.float32)
    np.maximum.at(best, S["owner"], sims)                       # คะแนนดีที่สุดต่อรูป (รูปหมู่ไม่ซ้ำ)
    idx = np.where(best >= min(max(min_similarity, 0.2), 0.8))[0]
    idx = idx[np.argsort(-best[idx])]
    results = [{**urls(str(S["names"][i])), "similarity": round(float(best[i]) * 100, 1)} for i in idx[:300]]   # แสดงสูงสุด 300 รูป
    return {"success": True, "faces_in_upload": len(faces), "total": int(len(idx)), "shown": len(results), "results": results}


for d in ("photos", "thumbs"):
    os.makedirs(f"data/{d}", exist_ok=True)
    app.mount(f"/{d}", StaticFiles(directory=f"data/{d}"), name=d)


@app.get("/")
def home():
    return FileResponse("index.html")