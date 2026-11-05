import asyncio, base64, hashlib, hmac, io, json, os, re, secrets, threading, time, urllib.request
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from insightface.app import FaceAnalysis
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

MODEL = os.getenv("MODEL_NAME", "buffalo_sc")
ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
SHEET_LOG_URL = os.getenv("SHEET_LOG_URL", "")
DELETE_SHEET_URL = os.getenv("DELETE_SHEET_URL", "") or SHEET_LOG_URL
EMB_URL = os.getenv("EMB_URL", "")
SELF_URL = os.getenv("KEEPALIVE_URL", "") or os.getenv("RENDER_EXTERNAL_URL", "")
KEEPALIVE_SEC = max(60, int(os.getenv("KEEPALIVE_SEC", "240")))
DELETE_MIN_SIM = float(os.getenv("DELETE_MIN_SIM", "0.42"))
TOKEN_SECRET = (os.getenv("TOKEN_SECRET") or secrets.token_hex(32)).encode()
TOKEN_TTL = 20 * 60
MAX_BYTES, RATE_PER_MIN = 10 * 1024 * 1024, 12
MAX_PARALLEL = int(os.getenv("MAX_PARALLEL", "2"))
TZ_TH = timezone(timedelta(hours=7))

S = {"fa": None, "emb": np.zeros((0, 512), np.int8), "inv": np.zeros(0, np.float32), "owner": np.zeros(0, np.int32),
     "names": np.array([], dtype=str), "index": {}, "src": "local"}
hits = defaultdict(deque)
dhits = defaultdict(deque)
used_tokens = {}
gate = threading.BoundedSemaphore(MAX_PARALLEL)


# ───────────────────────── เริ่มระบบ ─────────────────────────
def load_index():
    path = "data/embeddings.npz"
    if not os.path.exists(path) and EMB_URL:
        os.makedirs("data", exist_ok=True)
        urllib.request.urlretrieve(EMB_URL, path)
    if not os.path.exists(path):
        print("⚠️ ไม่พบ data/embeddings.npz — รัน build_index.py ก่อน", flush=True)
        return
    z = np.load(path)
    emb = z["embeddings"] if "embeddings" in z else z["emb"]
    inv = np.empty(len(emb), np.float32)
    for a in range(0, len(emb), 20000):
        n = np.linalg.norm(emb[a:a + 20000].astype(np.float32), axis=1)
        inv[a:a + 20000] = np.where(n > 0, 1.0 / np.maximum(n, 1e-9), 0.0)
    owner = z["owner"] if "owner" in z else np.arange(len(emb), dtype=np.int32)
    names = z["names"]
    if "model" in z and str(z["model"]) != MODEL:
        print(f"⚠️ embeddings สร้างจากโมเดล {z['model']} แต่ระบบใช้ {MODEL} — ผลค้นหาจะไม่แม่น", flush=True)
    S.update(emb=emb, inv=inv, owner=owner.astype(np.int32), names=names,
             index={str(n): i for i, n in enumerate(names)},
             src="drive")
    print(f"✅ พร้อมใช้งาน: {len(names)} รูป / {len(emb)} ใบหน้า / โมเดล {MODEL}", flush=True)


def load_all():
    fa = FaceAnalysis(name=MODEL, allowed_modules=["detection", "recognition"], providers=["CPUExecutionProvider"])
    fa.prepare(ctx_id=0, det_size=(640, 640))
    S["fa"] = fa
    load_index()


def _ping_self():
    try:
        urllib.request.urlopen(urllib.request.Request(SELF_URL.rstrip("/") + "/healthz", headers={"User-Agent": "stss-keepalive"}), timeout=15).read()
    except Exception as e:
        print(f"[keepalive] ping ไม่สำเร็จ: {e}", flush=True)


async def keepalive():
    while True:
        await asyncio.sleep(KEEPALIVE_SEC)
        await asyncio.to_thread(_ping_self)
        prune()


@asynccontextmanager
async def lifespan(_: FastAPI):
    task = None
    if SELF_URL and os.getenv("KEEPALIVE", "1") != "0":
        task = asyncio.create_task(keepalive())
        print(f"[keepalive] เปิดใช้งาน ทุก {KEEPALIVE_SEC}s → {SELF_URL}", flush=True)
    asyncio.get_running_loop().run_in_executor(None, load_all)
    try:
        yield
    finally:
        if task:
            task.cancel()


app = FastAPI(title="STSS STUDIO Face Search", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["*"], allow_headers=["*"])


# ───────────────────────── ตัวช่วย ─────────────────────────
def urls(n):
    if S["src"] == "drive":
        return {"thumb": f"https://drive.google.com/thumbnail?id={n}&sz=w480", "full": f"https://drive.google.com/thumbnail?id={n}&sz=w1600",
                "open": f"https://drive.google.com/file/d/{n}/view", "download": f"https://drive.google.com/uc?export=download&id={n}"}
    return {"thumb": f"/thumbs/{n}", "full": f"/photos/{n}", "open": f"/photos/{n}", "download": f"/photos/{n}"}


def score(q, rows=None):
    emb, inv = S["emb"], S["inv"]
    if rows is not None:
        return (emb[rows].astype(np.float32) @ q) * inv[rows]
    out = np.empty(len(emb), np.float32)
    for a in range(0, len(out), 20000):
        out[a:a + 20000] = (emb[a:a + 20000].astype(np.float32) @ q) * inv[a:a + 20000]
    return out


def client_ip(request: Request):
    return (request.headers.get("x-forwarded-for") or (request.client.host if request.client else "?")).split(",")[0].strip()


def limit(bucket, ip, n, window, msg):
    q, now = bucket[ip], time.time()
    while q and now - q[0] > window:
        q.popleft()
    if len(q) >= n:
        raise HTTPException(429, msg)
    q.append(now)


def guard(request: Request):
    ip = client_ip(request)
    limit(hits, ip, RATE_PER_MIN, 60, "ค้นหาถี่เกินไป รอ 1 นาทีแล้วลองใหม่")
    return ip


def prune():
    now = time.time()
    for bucket, win in ((hits, 60), (dhits, 3600)):
        for k in [k for k, q in bucket.items() if not q or now - q[-1] > win]:
            bucket.pop(k, None)
    for k in [k for k, exp in used_tokens.items() if exp < now]:
        used_tokens.pop(k, None)


# แก้ไข post_json ให้รองรับ Redirect 302 ของ Google Apps Script แบบไม่เสีย Payload
def post_json(url, row, timeout=10):
    data_bytes = json.dumps(row).encode("utf-8")
    curr_url = url
    for _ in range(3):
        req = urllib.request.Request(curr_url, data=data_bytes, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code in (301, 302, 303, 307, 308) and "Location" in e.headers:
                curr_url = e.headers["Location"]
                continue
            raise e


def send_to_sheet(row: dict):
    if not SHEET_LOG_URL:
        return
    try:
        post_json(SHEET_LOG_URL, row, timeout=8)
    except Exception as e:
        print(f"[sheet] ส่ง log เข้าชีตไม่สำเร็จ: {e}", flush=True)


def log(ip, thr, outcome, t0, bg: BackgroundTasks, faces=0, matches=0):
    ms = int((time.time() - t0) * 1000)
    print(f"[search] ip={ip} thr={thr} outcome={outcome} faces={faces} matches={matches} ms={ms}", flush=True)
    bg.add_task(send_to_sheet, {"type": "search", "ip": ip, "threshold": thr, "outcome": outcome, "faces": faces, "matches": matches, "ms": ms})


def read_image(file: UploadFile):
    raw = file.file.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "ไฟล์ใหญ่เกิน 10MB")
    try:
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    except Exception:
        raise HTTPException(400, "อ่านไฟล์รูปไม่ได้ ลองใช้ JPG, PNG หรือ WEBP")
    im.thumbnail((1280, 1280))
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def detect(bgr):
    if not gate.acquire(timeout=45):
        raise HTTPException(503, "ระบบกำลังมีผู้ใช้งานจำนวนมาก ลองใหม่อีกครั้งในอีกสักครู่")
    try:
        return S["fa"].get(bgr)
    finally:
        gate.release()


def need_ready():
    if S["fa"] is None:
        raise HTTPException(503, "ระบบกำลังเริ่มทำงาน ลองใหม่อีกครั้งในอีกสักครู่")


# ───────────────────────── ใบยืนยันใบหน้า (token) ─────────────────────────
def make_token(photo_id: str, sim: float) -> str:
    body = base64.urlsafe_b64encode(json.dumps({"p": photo_id, "s": round(sim, 3), "e": int(time.time()) + TOKEN_TTL, "n": secrets.token_hex(6)}).encode()).decode()
    return body + "." + hmac.new(TOKEN_SECRET, body.encode(), hashlib.sha256).hexdigest()


def read_token(tok: str):
    try:
        body, sig = tok.split(".", 1)
        if not hmac.compare_digest(sig, hmac.new(TOKEN_SECRET, body.encode(), hashlib.sha256).hexdigest()):
            return None
        d = json.loads(base64.urlsafe_b64decode(body.encode()))
        return d if d["e"] > time.time() else None
    except Exception:
        return None


def clean(s: str, n: int) -> str:
    s = re.sub(r"\s+", " ", (s or "").replace("\x00", "")).strip()[:n]
    return "'" + s if s[:1] in ("=", "+", "-", "@") else s


# ───────────────────────── API ─────────────────────────
@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/api/info")
def info():
    return {"ready": S["fa"] is not None and len(S["names"]) > 0, "photos": int(len(S["names"])), "model": MODEL}


@app.post("/api/search")
def search(request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...), min_similarity: float = Form(0.45)):
    ip = guard(request)
    t0 = time.time()
    need_ready()
    try:
        bgr = read_image(file)
    except HTTPException as e:
        log(ip, min_similarity, "too_large" if e.status_code == 413 else "bad_image", t0, background_tasks)
        raise
    if len(S["names"]) == 0:
        log(ip, min_similarity, "empty_db", t0, background_tasks)
        return {"success": False, "message": "ยังไม่มีรูปในคลัง กรุณาติดต่อผู้ดูแลระบบ"}
    faces = detect(bgr)
    if not faces:
        log(ip, min_similarity, "no_face", t0, background_tasks)
        return {"success": False, "message": "ไม่พบใบหน้าในรูป ลองใช้รูปที่เห็นหน้าตรง สว่าง และชัดเจน"}

    me = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    sims = score(me.normed_embedding.astype(np.float32))
    best = np.full(len(S["names"]), -1.0, np.float32)
    np.maximum.at(best, S["owner"], sims)
    idx = np.where(best >= min(max(min_similarity, 0.2), 0.8))[0]
    idx = idx[np.argsort(-best[idx])]
    results = [{"id": str(S["names"][i]), **urls(str(S["names"][i])), "similarity": round(min(float(best[i]), 1.0) * 100, 1)} for i in idx[:300]]
    log(ip, min_similarity, "ok", t0, background_tasks, faces=len(faces), matches=len(idx))
    return {"success": True, "faces_in_upload": len(faces), "total": int(len(idx)), "shown": len(results), "results": results}


@app.post("/api/delete/verify")
def delete_verify(request: Request, file: UploadFile = File(...), photo_id: str = Form(...)):
    ip = client_ip(request)
    limit(dhits, ip, 8, 600, "ตรวจใบหน้าถี่เกินไป รอสักครู่แล้วลองใหม่")
    need_ready()
    i = S["index"].get(photo_id)
    if i is None:
        raise HTTPException(404, "ไม่พบรูปนี้ในระบบ")
    faces = detect(read_image(file))
    if not faces:
        return {"success": False, "message": "ไม่พบใบหน้าในรูปที่สแกน ลองอยู่ในที่สว่าง มองกล้องตรง ๆ"}
    me = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    rows = np.where(S["owner"] == i)[0]
    sim = float(score(me.normed_embedding.astype(np.float32), rows).max()) if len(rows) else -1.0
    ok = sim >= DELETE_MIN_SIM
    print(f"[delete-verify] ip={ip} photo={photo_id} ok={ok}", flush=True)
    if not ok:
        return {"success": False, "message": "ใบหน้าที่สแกนไม่ตรงกับบุคคลในรูปนี้ ระบบรับคำขอเฉพาะผู้ที่ปรากฏอยู่ในรูปเท่านั้น"}
    return {"success": True, "token": make_token(photo_id, sim), "expires_in": TOKEN_TTL}


class DeleteReq(BaseModel):
    token: str
    name: str = Field(min_length=2, max_length=100)
    grade: str = Field(min_length=1, max_length=30)
    student_id: str = Field(min_length=3, max_length=20, pattern=r"^[0-9A-Za-z\-/]+$")
    reason: str = Field(min_length=10, max_length=600)


@app.post("/api/delete/request")
def delete_request(request: Request, body: DeleteReq):
    ip = client_ip(request)
    limit(dhits, ip, 8, 600, "ส่งคำขอถี่เกินไป รอสักครู่แล้วลองใหม่")
    if not DELETE_SHEET_URL:
        raise HTTPException(503, "ระบบรับคำขอลบรูปยังไม่เปิดใช้งาน กรุณาติดต่อผู้ดูแลระบบ")
    tok = read_token(body.token)
    if not tok or tok["n"] in used_tokens:
        raise HTTPException(400, "การยืนยันใบหน้าหมดอายุหรือถูกใช้ไปแล้ว กรุณาสแกนใบหน้าใหม่")
    pid = tok["p"]
    now = datetime.now(TZ_TH)
    ref = f"DEL-{now:%y%m%d}-{secrets.token_hex(2).upper()}"
    row = {"type": "delete_request", "ref": ref, "time": now.strftime("%Y-%m-%d %H:%M:%S"), "photo_id": pid,
           "photo_url": f"https://drive.google.com/file/d/{pid}/view", "name": clean(body.name, 100), "grade": clean(body.grade, 30),
           "student_id": clean(body.student_id, 20), "reason": clean(body.reason, 600), "face_match": round(tok["s"] * 100, 1),
           "ip": ip, "status": "รอพิจารณา"}
    for attempt in (1, 2):
        try:
            post_json(DELETE_SHEET_URL, row, timeout=15)
            break
        except Exception as e:
            print(f"[delete-request] ส่งเข้าชีตไม่สำเร็จ (ครั้งที่ {attempt}): {e}", flush=True)
            if attempt == 2:
                raise HTTPException(502, "ส่งคำขอไม่สำเร็จ กรุณาลองใหม่อีกครั้ง")
            time.sleep(1.5)
    used_tokens[tok["n"]] = tok["e"]
    print(f"[delete-request] ref={ref} photo={pid} ip={ip}", flush=True)
    return {"success": True, "ref": ref}


for d in ("photos", "thumbs"):
    os.makedirs(f"data/{d}", exist_ok=True)
    app.mount(f"/{d}", StaticFiles(directory=f"data/{d}"), name=d)


@app.get("/")
def home():
    return FileResponse("index.html", headers={"Cache-Control": "no-cache"})