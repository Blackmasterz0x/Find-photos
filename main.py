import asyncio, base64, hashlib, hmac, io, json, os, queue, re, secrets, threading, time, traceback, urllib.error, urllib.parse, urllib.request
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Literal

import cv2
import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
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
MAX_BYTES = 10 * 1024 * 1024
RATE_PER_MIN = int(os.getenv("RATE_PER_MIN", "12"))      # ค้นหา/นาที/IP (โรงเรียนใช้ Wi-Fi เดียวกัน IP ซ้ำกันได้ ปรับเพิ่มได้)
DELETE_RATE = int(os.getenv("DELETE_RATE", "8"))         # ขั้นตอนขอลบ/10 นาที/IP
MAX_PARALLEL = int(os.getenv("MAX_PARALLEL", "2"))
TZ_TH = timezone(timedelta(hours=7))

# ── รูปที่ถูกลบ (ซ่อนจากผลค้นหา) ──
REMOVED_URL = os.getenv("REMOVED_URL", "")               # URL (เช่น Apps Script ?action=removed) ที่คืนรายชื่อ photo_id ที่อนุมัติลบแล้ว
REMOVED_REFRESH_SEC = max(60, int(os.getenv("REMOVED_REFRESH_SEC", "300")))
AUTOHIDE = os.getenv("AUTOHIDE_BROKEN", "1") != "0"      # ซ่อนรูปที่ตรวจแล้วว่าเปิดจาก Drive ไม่ได้ (ปิดด้วย 0)
# ── แอดมิน ──
ADMIN_KEY = os.getenv("ADMIN_KEY", "")                   # รหัสผ่านแอดมิน (ไม่ตั้ง = ปิดทั้งระบบแอดมิน) และเป็น key ที่ Apps Script ใช้ตรวจ
ADMINS_FILE = os.getenv("ADMINS_FILE", "admins.json")    # ใบหน้าแอดมิน (สร้างด้วย enroll_admin.py) — หรือใส่ JSON ทั้งก้อนใน env ADMIN_FACES
ADMIN_REQUIRE_FACE = os.getenv("ADMIN_REQUIRE_FACE", "auto")   # auto = บังคับสแกนใบหน้าเมื่อมีใบหน้าลงทะเบียนแล้ว | 1 = บังคับเสมอ | 0 = ใช้รหัสผ่านอย่างเดียว
ADMIN_FACE_MIN_SIM = float(os.getenv("ADMIN_FACE_MIN_SIM", "0.5"))   # ความเหมือนขั้นต่ำ (เข้มกว่าการยืนยันขอลบรูป 0.42)
ADMIN_SESSION_HOURS = max(1, int(os.getenv("ADMIN_SESSION_HOURS", "8")))
NOTIFY_WEBHOOK_URL = os.getenv("NOTIFY_WEBHOOK_URL", "")  # (ไม่บังคับ) Discord/Slack webhook แจ้งเตือนเมื่อมีคำขอลบรูปใหม่
GONE_FILE = "data/gone.json"
IDX_PATH = "data/embeddings.npz"


def empty_ix():
    return {"emb": np.zeros((0, 512), np.int8), "inv": np.zeros(0, np.float32), "owner": np.zeros(0, np.int32),
            "names": np.array([], dtype=str), "index": {}, "hide": np.zeros(0, bool)}


# S["ix"] เป็น "ภาพนิ่งของฐานข้อมูล" ทั้งก้อน — โหลดใหม่แล้วสลับทีเดียว ระหว่างนั้นคำค้นหาที่กำลังทำงานจะไม่เห็นข้อมูลครึ่ง ๆ กลาง ๆ
S = {"fa": None, "ix": empty_ix(), "src": "local"}
hits = defaultdict(deque)
dhits = defaultdict(deque)
bhits = defaultdict(deque)
ahits = defaultdict(deque)
used_tokens = {}
gate = threading.BoundedSemaphore(MAX_PARALLEL)
verify_gate = threading.BoundedSemaphore(3)
GONE_LOCK = threading.Lock()
GONE_AUTO, GONE_REMOTE, GONE_MANUAL = set(), set(), set()   # AUTO = ระบบตรวจเจอว่าเปิดไม่ได้เอง, REMOTE = มาจากชีต (REMOVED_URL), MANUAL = แอดมินสั่งซ่อน/อนุมัติผ่านหน้า /admin
checked = {}                                             # photo_id -> เวลาที่ตรวจล่าสุด (กันตรวจซ้ำถี่ ๆ)
STARTED = time.time()


# ───────────────────────── รูปที่ถูกลบ ─────────────────────────
def gone_ids():
    return GONE_AUTO | GONE_REMOTE | GONE_MANUAL


def build_mask(names):
    gone = gone_ids()
    return np.fromiter((str(n) in gone for n in names), bool, count=len(names)) if gone else np.zeros(len(names), bool)


def refresh_mask():
    ix = S["ix"]
    ix["hide"] = build_mask(ix["names"])


def load_gone_file():
    try:
        with open(GONE_FILE, encoding="utf-8") as f:
            d = json.load(f)
        GONE_AUTO.update(str(x) for x in d.get("auto", []))
        GONE_MANUAL.update(str(x) for x in d.get("manual", []))
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"[gone] อ่าน {GONE_FILE} ไม่สำเร็จ: {e}", flush=True)


def save_gone_file():
    try:
        os.makedirs("data", exist_ok=True)
        tmp = GONE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"auto": sorted(GONE_AUTO), "manual": sorted(GONE_MANUAL)}, f)
        os.replace(tmp, GONE_FILE)
    except Exception as e:
        print(f"[gone] บันทึกไม่สำเร็จ: {e}", flush=True)


DRIVE_ID = re.compile(r"[A-Za-z0-9_-]{20,}")


def parse_ids(raw: str):
    """รับได้ทั้ง JSON {"removed":[...]} / [...] / ข้อความบรรทัดละ id / ลิงก์ Drive — ดึงเฉพาะสิ่งที่หน้าตาเป็น Drive ID"""
    return set(DRIVE_ID.findall(raw))


def fetch_removed():
    if not REMOVED_URL:
        return
    try:
        with urllib.request.urlopen(urllib.request.Request(REMOVED_URL, headers={"User-Agent": "stss-removed"}), timeout=25) as r:
            raw = r.read(4_000_000).decode("utf-8", "replace")
        if raw.lstrip()[:1] == "<":                       # ได้หน้า HTML (เช่น Apps Script error/ล็อกอิน) ห้ามเอามาเขียนทับรายการเดิม
            raise ValueError("ได้หน้า HTML แทนรายการ id")
        ids = parse_ids(raw)
    except Exception as e:
        print(f"[removed] ดึงรายการรูปที่ลบไม่สำเร็จ (ใช้รายการเดิมต่อ): {e}", flush=True)
        return
    with GONE_LOCK:
        changed = ids != GONE_REMOTE
        GONE_REMOTE.clear(); GONE_REMOTE.update(ids)
        if changed:
            refresh_mask()
    if changed:
        print(f"[removed] อัปเดตรายการรูปที่ซ่อน: {len(ids)} รูป", flush=True)


def drive_state(pid: str):
    """True = เปิดได้ / False = เปิดไม่ได้แน่ ๆ (ถูกลบ/ไม่แชร์) / None = ไม่แน่ใจ (Google จำกัดความถี่, เน็ตมีปัญหา)"""
    req = urllib.request.Request(f"https://drive.google.com/thumbnail?id={pid}&sz=w64", headers={"User-Agent": "Mozilla/5.0 (stss-verify)"})
    try:
        with urllib.request.urlopen(req, timeout=12) as r:
            return str(r.headers.get("Content-Type", "")).startswith("image/")   # 200 แต่เป็น HTML = หน้าล็อกอิน/ไม่พบไฟล์
    except urllib.error.HTTPError as e:
        return False if e.code in (400, 401, 403, 404, 410) else None
    except Exception:
        return None


def confirm_dead(pid: str):
    """ตรวจ 2 รอบห่างกัน 1.5 วินาที ต้องได้ "เปิดไม่ได้" ทั้งคู่ถึงจะถือว่าถูกลบ (กันซ่อนรูปดีเพราะ Google สะดุดชั่วคราว)"""
    with verify_gate:
        if drive_state(pid) is not False:
            return False
        time.sleep(1.5)
        return drive_state(pid) is False


# ───────────────────────── เริ่มระบบ ─────────────────────────
def download_embeddings():
    """โหลดไฟล์ฐานข้อมูลลงไฟล์ชั่วคราวก่อน ตรวจว่าอ่านได้ แล้วค่อยสลับ — ถ้าโหลดค้างครึ่งทาง ไฟล์เดิมจะไม่พัง"""
    os.makedirs("data", exist_ok=True)
    tmp = IDX_PATH + ".tmp"
    urllib.request.urlretrieve(EMB_URL, tmp)
    with np.load(tmp) as z:
        if "names" not in z or not ("embeddings" in z or "emb" in z):
            raise ValueError("ไฟล์ที่ดาวน์โหลดไม่ใช่ embeddings.npz ที่ถูกต้อง (EMB_URL อาจได้หน้า HTML)")
    os.replace(tmp, IDX_PATH)


def load_index(force_download=False):
    if (force_download or not os.path.exists(IDX_PATH)) and EMB_URL:
        download_embeddings()
    if not os.path.exists(IDX_PATH):
        print("⚠️ ไม่พบ data/embeddings.npz — รัน build_index.py ก่อน", flush=True)
        return False
    z = np.load(IDX_PATH)
    emb = z["embeddings"] if "embeddings" in z else z["emb"]
    inv = np.empty(len(emb), np.float32)
    for a in range(0, len(emb), 20000):
        n = np.linalg.norm(emb[a:a + 20000].astype(np.float32), axis=1)
        inv[a:a + 20000] = np.where(n > 0, 1.0 / np.maximum(n, 1e-9), 0.0)
    owner = z["owner"] if "owner" in z else np.arange(len(emb), dtype=np.int32)
    names = z["names"]
    if "model" in z and str(z["model"]) != MODEL:
        print(f"⚠️ embeddings สร้างจากโมเดล {z['model']} แต่ระบบใช้ {MODEL} — ผลค้นหาจะไม่แม่น", flush=True)
    ix = {"emb": emb, "inv": inv, "owner": owner.astype(np.int32), "names": names,
          "index": {str(n): i for i, n in enumerate(names)}, "hide": None}
    with GONE_LOCK:                                    # สร้างหน้ากากซ่อนรูปกับสลับฐานข้อมูลใต้ล็อกเดียวกัน กันพลาดช่วงที่มีคนเพิ่งซ่อนรูปพอดี
        ix["hide"] = build_mask(names)
        S["ix"], S["src"] = ix, "drive"               # สลับทีเดียว
    print(f"✅ พร้อมใช้งาน: {len(names)} รูป / {len(emb)} ใบหน้า / ซ่อนอยู่ {int(ix['hide'].sum())} รูป / โมเดล {MODEL}", flush=True)
    return True


def load_all():
    try:
        fa = FaceAnalysis(name=MODEL, allowed_modules=["detection", "recognition"], providers=["CPUExecutionProvider"])
        fa.prepare(ctx_id=0, det_size=(640, 640))
        S["fa"] = fa
    except Exception:
        print("❌ โหลดโมเดลไม่สำเร็จ:\n" + traceback.format_exc(), flush=True)
        return
    load_gone_file()
    fetch_removed()
    for attempt in range(1, 7):                        # เดิมถ้าดาวน์โหลดฐานข้อมูลพลาดครั้งเดียว ระบบจะค้างว่า "กำลังเริ่มทำงาน" ตลอดไปโดยไม่มี log
        try:
            load_index()
            return
        except Exception as e:
            print(f"❌ โหลดฐานข้อมูลไม่สำเร็จ (ครั้งที่ {attempt}/6): {e}", flush=True)
            time.sleep(min(60, 5 * attempt))


def _ping_self():
    try:
        urllib.request.urlopen(urllib.request.Request(SELF_URL.rstrip("/") + "/healthz", headers={"User-Agent": "stss-keepalive"}), timeout=15).read()
    except Exception as e:
        print(f"[keepalive] ping ไม่สำเร็จ: {e}", flush=True)


async def maintenance():
    last_ping = last_rm = time.time()
    while True:
        await asyncio.sleep(30)
        now = time.time()
        if SELF_URL and os.getenv("KEEPALIVE", "1") != "0" and now - last_ping >= KEEPALIVE_SEC:
            last_ping = now
            await asyncio.to_thread(_ping_self)
        if REMOVED_URL and now - last_rm >= REMOVED_REFRESH_SEC:
            last_rm = now
            await asyncio.to_thread(fetch_removed)
        prune()


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not os.getenv("TOKEN_SECRET"):
        print("⚠️ ไม่ได้ตั้ง TOKEN_SECRET — ทุกครั้งที่เซิร์ฟเวอร์รีสตาร์ท ใบยืนยันใบหน้าของผู้ใช้ที่กำลังกรอกแบบฟอร์มลบรูปจะใช้ไม่ได้", flush=True)
    load_admins()
    if not ADMIN_KEY:
        print("ℹ️ ไม่ได้ตั้ง ADMIN_KEY — ระบบแอดมินปิดอยู่", flush=True)
    task = asyncio.create_task(maintenance())
    if SELF_URL and os.getenv("KEEPALIVE", "1") != "0":
        print(f"[keepalive] เปิดใช้งาน ทุก {KEEPALIVE_SEC}s → {SELF_URL}", flush=True)
    asyncio.get_running_loop().run_in_executor(None, load_all)
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(title="STSS STUDIO Face Search", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["*"], allow_headers=["*"])


# ───────────────────────── ตัวช่วย ─────────────────────────
def urls(n):
    if S["src"] == "drive":
        return {"thumb": f"https://drive.google.com/thumbnail?id={n}&sz=w480", "full": f"https://drive.google.com/thumbnail?id={n}&sz=w1600",
                "open": f"https://drive.google.com/file/d/{n}/view", "download": f"https://drive.google.com/uc?export=download&id={n}"}
    return {"thumb": f"/thumbs/{n}", "full": f"/photos/{n}", "open": f"/photos/{n}", "download": f"/photos/{n}"}


def score(ix, q, rows=None):
    emb, inv = ix["emb"], ix["inv"]
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
    # list(...) ก่อนวน: prune ทำงานในลูปหลัก ขณะที่ endpoint ทำงานในเธรดอื่น ถ้าวนตรง ๆ แล้วมีคนเพิ่ม key พอดีจะ error "dictionary changed size"
    for bucket, win in ((hits, 60), (dhits, 3600), (bhits, 3600), (ahits, 3600)):
        for k in [k for k, q in list(bucket.items()) if not q or now - q[-1] > win]:
            bucket.pop(k, None)
    for k in [k for k, exp in list(used_tokens.items()) if exp < now]:
        used_tokens.pop(k, None)
    for k in [k for k, t in list(checked.items()) if now - t > 900]:
        checked.pop(k, None)


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
@app.api_route("/healthz", methods=["GET", "HEAD"])      # HEAD ด้วย: บอทเช็กเว็บฟรี (เช่น UptimeRobot) ส่ง HEAD ถ้าไม่รองรับจะได้ 405
def healthz():
    return {"ok": True}


@app.get("/api/info")
def info():
    ix = S["ix"]
    return {"ready": S["fa"] is not None and len(ix["names"]) > 0, "photos": int(len(ix["names"]) - ix["hide"].sum()), "model": MODEL}


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
    ix = S["ix"]                                       # จับภาพนิ่งไว้ใช้ตลอดคำขอนี้
    if len(ix["names"]) == 0:
        log(ip, min_similarity, "empty_db", t0, background_tasks)
        return {"success": False, "message": "ยังไม่มีรูปในคลัง กรุณาติดต่อผู้ดูแลระบบ"}
    faces = detect(bgr)
    if not faces:
        log(ip, min_similarity, "no_face", t0, background_tasks)
        return {"success": False, "message": "ไม่พบใบหน้าในรูป ลองใช้รูปที่เห็นหน้าตรง สว่าง และชัดเจน"}

    me = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    sims = score(ix, me.normed_embedding.astype(np.float32))
    best = np.full(len(ix["names"]), -1.0, np.float32)
    np.maximum.at(best, ix["owner"], sims)
    best[ix["hide"]] = -1.0                             # รูปที่ถูกลบ/เปิดไม่ได้ ไม่ถูกส่งกลับเป็น "เพลท" เปล่า ๆ อีก
    idx = np.where(best >= min(max(min_similarity, 0.2), 0.8))[0]
    idx = idx[np.argsort(-best[idx])]
    results = [{"id": str(ix["names"][i]), **urls(str(ix["names"][i])), "similarity": round(min(float(best[i]), 1.0) * 100, 1)} for i in idx[:300]]
    log(ip, min_similarity, "ok", t0, background_tasks, faces=len(faces), matches=len(idx))
    return {"success": True, "faces_in_upload": len(faces), "total": int(len(idx)), "shown": len(results), "results": results}


class BrokenReq(BaseModel):
    id: str = Field(min_length=10, max_length=80)


@app.post("/api/broken")
def broken(request: Request, body: BrokenReq, background_tasks: BackgroundTasks):
    """หน้าเว็บแจ้งว่ารูปนี้โหลดไม่ขึ้น — เซิร์ฟเวอร์ไม่เชื่อผู้ใช้ แต่ไปตรวจกับ Drive เองก่อน ถ้าเปิดไม่ได้จริงถึงจะซ่อน"""
    ip = client_ip(request)
    limit(bhits, ip, 60, 600, "แจ้งถี่เกินไป")
    pid = body.id
    if pid not in S["ix"]["index"]:                     # รับเฉพาะ id ที่มีในฐานข้อมูล (กันถูกใช้ยิงลิงก์มั่ว)
        return {"dead": False}
    if pid in gone_ids():
        return {"dead": True}
    if not AUTOHIDE or time.time() - checked.get(pid, 0) < 60:
        return {"dead": False}
    checked[pid] = time.time()
    if not confirm_dead(pid):
        return {"dead": False}
    with GONE_LOCK:
        GONE_AUTO.add(pid)
        refresh_mask()
    save_gone_file()
    print(f"[broken] ซ่อนรูปที่เปิดจาก Drive ไม่ได้: {pid}", flush=True)
    background_tasks.add_task(send_to_sheet, {"type": "photo_gone", "photo_id": pid, "photo_url": f"https://drive.google.com/file/d/{pid}/view", "ip": ip})
    return {"dead": True}


# ───────────────────────── แอดมิน ─────────────────────────
# ล็อกอิน 2 ขั้น
#   (1) POST /api/admin/login  {password}  → ผ่านแล้วได้ "pre-token" อายุ 5 นาที
#   (2) POST /api/admin/face   รูปจากกล้อง + หัว X-Pre-Token → ใบหน้าตรงกับใครใน admins.json ก็ได้ "session-token" ที่ฝังชื่อคนนั้นไว้
# ถ้ายังไม่มีใครลงทะเบียนใบหน้า (admins.json ว่าง) = เข้าด้วยรหัสผ่านอย่างเดียว (ชื่อในบันทึก = env ADMIN_NAME, ค่าเริ่มต้น adminxkhetnxkon_)
# ทุก endpoint ที่เหลือใช้หัว X-Admin-Token (session-token) และบันทึกลงชีตว่า "ใคร ทำอะไร เมื่อไหร่ จาก IP ไหน"
ADMINS_STATE = {"list": [], "broken": False}
SESSION_TTL = ADMIN_SESSION_HOURS * 3600
NO_FACE_NAME = (os.getenv("ADMIN_NAME", "adminxkhetnxkon_").strip() or "admin")[:60]   # ชื่อที่แสดง/บันทึกลงชีต ตอนเข้าด้วยรหัสผ่านอย่างเดียว (ยังไม่ได้ลงทะเบียนใบหน้า)


def sign_tok(d: dict) -> str:
    body = base64.urlsafe_b64encode(json.dumps(d, ensure_ascii=False).encode()).decode()
    return body + "." + hmac.new(TOKEN_SECRET, body.encode(), hashlib.sha256).hexdigest()


def unsign_tok(tok: str, kind: str):
    try:
        body, sig = (tok or "").split(".", 1)
        if not hmac.compare_digest(sig, hmac.new(TOKEN_SECRET, body.encode(), hashlib.sha256).hexdigest()):
            return None
        d = json.loads(base64.urlsafe_b64decode(body.encode()))
        return d if d.get("k") == kind and d["e"] > time.time() else None
    except Exception:
        return None


def make_session(name: str, aid: str, role: str, mode: str) -> str:
    return sign_tok({"k": "ses", "name": name, "aid": aid, "role": role, "m": mode, "e": int(time.time()) + SESSION_TTL, "n": secrets.token_hex(6)})


def load_admins():
    """อ่านใบหน้าแอดมินจาก env ADMIN_FACES (JSON) หรือไฟล์ admins.json — รูปแบบดู admins.example.json / enroll_admin.py"""
    raw, src = os.getenv("ADMIN_FACES", "").strip(), "env ADMIN_FACES"
    if not raw and os.path.exists(ADMINS_FILE):
        src = ADMINS_FILE
        try:
            with open(ADMINS_FILE, encoding="utf-8") as f:
                raw = f.read()
        except Exception as e:
            print(f"❌ อ่าน {ADMINS_FILE} ไม่ได้: {e}", flush=True)
            ADMINS_STATE.update(list=[], broken=True)
            return
    if not raw.strip():
        ADMINS_STATE.update(list=[], broken=False)
        return
    try:
        data = json.loads(raw)
        out = []
        for e in data.get("admins", []):
            emb = np.asarray(e.get("embeddings") or [], np.float32)
            if emb.ndim != 2 or emb.shape[1] != 512 or len(emb) == 0:
                print(f"⚠️ แอดมิน {e.get('name', '?')}: รูปแบบรหัสใบหน้าไม่ถูกต้อง (ต้องเป็นรายการของเวกเตอร์ 512 ค่า) — ข้าม", flush=True); continue
            n = np.linalg.norm(emb, axis=1, keepdims=True)
            if (n == 0).any():
                print(f"⚠️ แอดมิน {e.get('name', '?')}: มีรหัสใบหน้าเป็นศูนย์ทั้งชุด (ยังเป็นค่าตัวอย่าง?) — ข้าม", flush=True); continue
            out.append({"id": str(e.get("id") or e.get("name")), "name": str(e["name"]), "role": str(e.get("role", "")), "emb": emb / n})
        if data.get("admins") and not out:
            raise ValueError("มีรายชื่อแอดมินแต่ไม่มีใครใช้งานได้")
        ADMINS_STATE.update(list=out, broken=False)
        print(f"✅ ใบหน้าแอดมิน: {len(out)} คน ({src}): {', '.join(a['name'] for a in out) or '-'}", flush=True)
    except Exception as e:
        # ไฟล์พัง/อ่านไม่ได้ = ปิดประตู (ห้ามถอยไปใช้รหัสผ่านอย่างเดียวเงียบ ๆ)
        print(f"❌ ใบหน้าแอดมินใช้ไม่ได้ ({src}): {e} — ล็อกอินแอดมินจะถูกปิดจนกว่าจะแก้", flush=True)
        ADMINS_STATE.update(list=[], broken=True)


def face_required() -> bool:
    mode = ADMIN_REQUIRE_FACE
    return mode == "1" or (mode != "0" and (bool(ADMINS_STATE["list"]) or ADMINS_STATE["broken"]))


def match_admin(q):
    best, who = -1.0, None
    for a in ADMINS_STATE["list"]:
        s = float((a["emb"] @ q).max())
        if s > best:
            best, who = s, a
    return who, best


def too_many_fails(ip: str) -> bool:
    q, now = ahits[ip], time.time()
    while q and now - q[0] > 600:
        q.popleft()
    return len(q) >= 10


def admin_auth(request: Request, token: str):
    """ตรวจ session-token — คืนข้อมูลคนที่ล็อกอิน {name, role, ip, mode}"""
    d = unsign_tok(token, "ses")
    if not d or d["n"] in used_tokens or (face_required() and d.get("m") != "face"):
        raise HTTPException(401, "เซสชันหมดอายุหรือยังไม่ได้เข้าสู่ระบบ")
    return {"name": d["name"], "aid": d.get("aid", ""), "role": d.get("role", ""), "mode": d.get("m", ""), "ip": client_ip(request), "n": d["n"], "e": d["e"]}


# ── บันทึกการใช้งานแอดมิน (audit log) ──
AUDIT_BUF = deque(maxlen=300)
AUDIT_Q = queue.Queue(maxsize=1000)
AUDIT_STATE = {"sent": 0, "failed": 0, "last_error": "", "worker": False}
AUDIT_LOCK = threading.Lock()


def _audit_worker():
    while True:
        row = AUDIT_Q.get()
        if not DELETE_SHEET_URL:
            continue
        err = ""
        for attempt in (1, 2, 3):
            try:
                post_json(DELETE_SHEET_URL, {**row, "key": ADMIN_KEY}, timeout=20)
                AUDIT_STATE["sent"] += 1
                err = ""
                break
            except Exception as e:
                err = str(e)
                time.sleep(2 * attempt)
        if err:
            AUDIT_STATE["failed"] += 1
            AUDIT_STATE["last_error"] = err[:200]
            print(f"[audit] ส่งเข้าชีตไม่สำเร็จ: {err}", flush=True)


def audit(who: str, ip: str, action: str, detail: str = "", ok: bool = True):
    """เขียนทุกการกระทำของแอดมิน: เก็บในหน่วยความจำ + ไฟล์ data/admin_log.jsonl + ส่งเข้าชีตเบื้องหลัง (ไม่ทำให้หน้าเว็บช้า)"""
    row = {"type": "admin_log", "time": datetime.now(TZ_TH).strftime("%Y-%m-%d %H:%M:%S"), "admin": who or "-", "ip": ip, "action": action,
           "detail": str(detail)[:400], "result": "สำเร็จ" if ok else "ล้มเหลว"}
    AUDIT_BUF.appendleft(row)
    print(f"[audit] {row['admin']} | {action} | {row['detail']} | {row['result']} | {ip}", flush=True)
    try:
        os.makedirs("data", exist_ok=True)
        with open("data/admin_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass
    try:
        AUDIT_Q.put_nowait(row)
    except queue.Full:
        AUDIT_STATE["failed"] += 1
    with AUDIT_LOCK:
        if not AUDIT_STATE["worker"]:
            AUDIT_STATE["worker"] = True
            threading.Thread(target=_audit_worker, daemon=True).start()


def notify(text: str):
    """แจ้งเตือนเข้า Discord/Slack-compatible webhook (ถ้าตั้ง NOTIFY_WEBHOOK_URL) — เบื้องหลัง ไม่รอผล"""
    if not NOTIFY_WEBHOOK_URL:
        return

    def run():
        try:
            data = json.dumps({"content": text[:1900], "text": text[:3000], "allowed_mentions": {"parse": []}}).encode()
            urllib.request.urlopen(urllib.request.Request(NOTIFY_WEBHOOK_URL, data=data, method="POST",
                                                          headers={"Content-Type": "application/json", "User-Agent": "stss-notify"}), timeout=10).read()
        except Exception as e:
            print(f"[notify] ส่งแจ้งเตือนไม่สำเร็จ: {e}", flush=True)
    threading.Thread(target=run, daemon=True).start()


# ── ล็อกอิน ──
class LoginReq(BaseModel):
    password: str = Field(min_length=1, max_length=200)


@app.post("/api/admin/login")
def admin_login(request: Request, body: LoginReq):
    ip = client_ip(request)
    if too_many_fails(ip):
        audit("-", ip, "ล็อกอิน", "ถูกบล็อกชั่วคราว (ใส่ผิดเกิน 10 ครั้งใน 10 นาที)", False)
        raise HTTPException(429, "ใส่ผิดบ่อยเกินไป รอ 10 นาทีแล้วลองใหม่")
    if not ADMIN_KEY:
        raise HTTPException(503, "ยังไม่ได้ตั้งค่า ADMIN_KEY บนเซิร์ฟเวอร์")
    if not hmac.compare_digest(body.password.encode(), ADMIN_KEY.encode()):
        ahits[ip].append(time.time())
        audit("-", ip, "ล็อกอินด้วยรหัสผ่าน", "รหัสผ่านไม่ถูกต้อง", False)
        raise HTTPException(403, "รหัสผ่านไม่ถูกต้อง")
    if face_required():
        if not ADMINS_STATE["list"]:
            audit("-", ip, "ล็อกอินด้วยรหัสผ่าน", "เปิดการสแกนใบหน้าแต่ไม่มีใบหน้าแอดมินที่ใช้งานได้ (ตรวจ admins.json)", False)
            raise HTTPException(503, "เปิดการสแกนใบหน้าไว้ แต่ยังไม่มีใบหน้าแอดมินที่ลงทะเบียน/ไฟล์ admins.json ใช้ไม่ได้ — ดู README ข้อ 8")
        audit("-", ip, "ล็อกอินด้วยรหัสผ่าน", "ผ่าน — รอสแกนใบหน้า", True)
        return {"step": "face", "pre": sign_tok({"k": "pre", "e": int(time.time()) + 300, "n": secrets.token_hex(6)}), "expires_in": 300}
    audit(NO_FACE_NAME, ip, "เข้าสู่ระบบ", "ด้วยรหัสผ่านอย่างเดียว (ยังไม่ได้เปิดการสแกนใบหน้า)", True)
    return {"step": "done", "token": make_session(NO_FACE_NAME, "", "", "pw"), "name": NO_FACE_NAME, "role": "", "mode": "pw", "expires_in": SESSION_TTL}


@app.post("/api/admin/face")
def admin_face(request: Request, file: UploadFile = File(...), x_pre_token: str = Header("")):
    ip = client_ip(request)
    if too_many_fails(ip):
        raise HTTPException(429, "สแกนไม่ผ่านบ่อยเกินไป รอ 10 นาทีแล้วลองใหม่")
    pre = unsign_tok(x_pre_token, "pre")
    if not pre or pre["n"] in used_tokens:
        raise HTTPException(401, "หมดเวลายืนยันรหัสผ่าน กรุณากรอกรหัสผ่านใหม่")
    need_ready()
    if not ADMINS_STATE["list"]:
        raise HTTPException(503, "ไม่มีใบหน้าแอดมินที่ลงทะเบียน")
    faces = detect(read_image(file))
    if not faces:
        return {"success": False, "message": "ไม่พบใบหน้าในภาพ ลองอยู่ในที่สว่างและมองกล้องตรง ๆ"}
    me = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    who, sim = match_admin(me.normed_embedding.astype(np.float32))
    if not who or sim < ADMIN_FACE_MIN_SIM:
        ahits[ip].append(time.time())
        audit("-", ip, "สแกนใบหน้า", f"ไม่ผ่าน — ใกล้เคียงที่สุด {max(sim, 0) * 100:.0f}% (ต้อง ≥ {ADMIN_FACE_MIN_SIM * 100:.0f}%)", False)
        return {"success": False, "message": "ใบหน้าไม่ตรงกับแอดมินที่ลงทะเบียนไว้"}
    used_tokens[pre["n"]] = pre["e"]
    audit(who["name"], ip, "เข้าสู่ระบบ", f"รหัสผ่าน + สแกนใบหน้าผ่าน ({sim * 100:.0f}%)", True)
    return {"success": True, "token": make_session(who["name"], who["id"], who["role"], "face"), "name": who["name"], "role": who["role"], "mode": "face", "expires_in": SESSION_TTL}


class AuditReq(BaseModel):
    action: Literal["logout", "export_csv", "export_removed"]


AUDIT_LABEL = {"logout": "ออกจากระบบ", "export_csv": "ดาวน์โหลดประวัติการค้นหา (CSV)", "export_removed": "ดาวน์โหลด removed.txt"}


@app.post("/api/admin/audit")
def admin_audit(request: Request, body: AuditReq, x_admin_token: str = Header("")):
    """บันทึกการกระทำที่เกิดในเบราว์เซอร์อย่างเดียว (ออกจากระบบ/ดาวน์โหลดไฟล์) — จำกัดเฉพาะรายการที่กำหนด กันเขียนข้อความปลอม"""
    a = admin_auth(request, x_admin_token)
    if body.action == "logout":
        used_tokens[a["n"]] = a["e"]                     # เพิกถอนเซสชันนี้ทันที
    audit(a["name"], a["ip"], AUDIT_LABEL[body.action])
    return {"ok": True}


def sheet_call(params: dict):
    """เรียก Apps Script ฝั่งแอดมิน (GET ตามด้วย ?action=...&key=ADMIN_KEY) — ตัวสคริปต์ต้องตรวจ key เดียวกันนี้"""
    if not DELETE_SHEET_URL:
        raise HTTPException(503, "ยังไม่ได้ตั้งค่า DELETE_SHEET_URL / SHEET_LOG_URL")
    url = DELETE_SHEET_URL + ("&" if "?" in DELETE_SHEET_URL else "?") + urllib.parse.urlencode({**params, "key": ADMIN_KEY})
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "stss-admin"}), timeout=30) as r:
            raw = r.read(5_000_000).decode("utf-8", "replace")
        d = json.loads(raw)
    except Exception as e:
        raise HTTPException(502, f"คุยกับ Apps Script ไม่สำเร็จ: {e}")
    if not isinstance(d, dict) or d.get("ok") is False:
        raise HTTPException(502, f"Apps Script ตอบกลับว่า: {(d or {}).get('error', 'ผิดพลาด') if isinstance(d, dict) else 'รูปแบบไม่ถูกต้อง'}")
    return d


def photo_row(pid: str, **extra):
    u = urls(pid)
    return {"id": pid, "thumb": u["thumb"], "open": f"https://drive.google.com/file/d/{pid}/view", "in_index": pid in S["ix"]["index"], **extra}


def drive_link(pid: str) -> str:
    return f"https://drive.google.com/file/d/{pid}/view"


@app.get("/api/admin/stats")
def admin_stats(request: Request, x_admin_token: str = Header("")):
    a = admin_auth(request, x_admin_token)               # เรียกถี่ (รีเฟรชตัวเลข) จึงไม่บันทึก audit
    ix = S["ix"]
    return {"me": {"name": a["name"], "role": a["role"], "mode": a["mode"], "expires_at": a["e"]},
            "ready": S["fa"] is not None and len(ix["names"]) > 0, "model": MODEL, "uptime_sec": int(time.time() - STARTED),
            "photos": int(len(ix["names"])), "faces": int(len(ix["emb"])), "searchable": int(np.unique(ix["owner"]).size),
            "hidden": {"auto": len(GONE_AUTO), "sheet": len(GONE_REMOTE), "manual": len(GONE_MANUAL), "masked_in_index": int(ix["hide"].sum())},
            "face_login": {"required": face_required(), "admins": len(ADMINS_STATE["list"]), "broken": ADMINS_STATE["broken"]},
            "audit": {"sent": AUDIT_STATE["sent"], "failed": AUDIT_STATE["failed"], "queued": AUDIT_Q.qsize(), "last_error": AUDIT_STATE["last_error"]},
            "config": {"token_secret": bool(os.getenv("TOKEN_SECRET")), "emb_url": bool(EMB_URL), "removed_url": bool(REMOVED_URL), "notify": bool(NOTIFY_WEBHOOK_URL),
                       "delete_sheet_url": bool(DELETE_SHEET_URL), "keepalive": bool(SELF_URL) and os.getenv("KEEPALIVE", "1") != "0", "autohide": AUTOHIDE}}


@app.get("/api/admin/hidden")
def admin_hidden(request: Request, x_admin_token: str = Header("")):
    a = admin_auth(request, x_admin_token)
    rows = [photo_row(pid, source=src) for src, s in (("manual", GONE_MANUAL), ("auto", GONE_AUTO), ("sheet", GONE_REMOTE)) for pid in sorted(s)]
    audit(a["name"], a["ip"], "ดูรายการรูปที่ถูกซ่อน", f"{len(rows)} รายการ")
    return {"rows": rows[:1000], "total": len(rows)}


class HideReq(BaseModel):
    ids: list[str] = Field(min_length=1, max_length=500)
    trash: bool = False                                  # True = ย้ายไฟล์ใน Drive เข้าถังขยะด้วย (ผ่าน Apps Script) ไม่ใช่แค่ซ่อนจากการค้นหา


@app.post("/api/admin/hide")
def admin_hide(request: Request, body: HideReq, x_admin_token: str = Header("")):
    """รับ id หรือลิงก์ Drive หลายอันได้ (วางทั้งก้อน) — ซ่อนเฉพาะที่มีในฐานข้อมูล"""
    a = admin_auth(request, x_admin_token)
    want = parse_ids(" ".join(body.ids))
    known = sorted(p for p in want if p in S["ix"]["index"])
    what = "ลบจาก Drive + ซ่อน" if body.trash else "ซ่อนจากการค้นหา"
    if body.trash and len(known) > 20:
        audit(a["name"], a["ip"], what, f"ปฏิเสธ: เลือก {len(known)} รูป เกินครั้งละ 20", False)
        raise HTTPException(400, "ลบจาก Drive ได้ครั้งละไม่เกิน 20 รูป")
    trashed, failed = [], []
    if body.trash:
        for pid in known:
            try:
                sheet_call({"action": "trash", "id": pid}); trashed.append(pid)
            except HTTPException as e:
                failed.append({"id": pid, "error": str(e.detail)})
    with GONE_LOCK:                                      # ซ่อนเสมอ แม้ลบไฟล์ไม่สำเร็จ — ผู้ดูแลสั่งให้รูปหายแล้ว ไม่ควรโผล่ในผลค้นหาต่อ
        GONE_MANUAL.update(known)
        refresh_mask()
    save_gone_file()
    audit(a["name"], a["ip"], what, f"{len(known)} รูป รหัส: {', '.join(known[:4])}{' …' if len(known) > 4 else ''}"
          + (f" | ลบไฟล์ไม่สำเร็จ {len(failed)}" if failed else "") + (f" | ไม่พบในระบบ {len(want) - len(known)}" if len(want) != len(known) else ""), not failed)
    return {"hidden": len(known), "unknown": sorted(want - set(known)), "trashed": trashed, "trash_failed": failed}


@app.get("/api/admin/history")
def admin_history(request: Request, limit: int = 500, x_admin_token: str = Header("")):
    """ประวัติการค้นหาจากชีต (บันทึกโดย send_to_sheet) — คืน {headers, rows} ตามหัวตารางจริงในชีต"""
    a = admin_auth(request, x_admin_token)
    try:
        d = sheet_call({"action": "history", "limit": max(1, min(limit, 1000))})
    except HTTPException as e:
        audit(a["name"], a["ip"], "ดูประวัติการค้นหา", str(e.detail), False)
        raise
    audit(a["name"], a["ip"], "ดูประวัติการค้นหา", f"{len(d.get('rows', []))} แถว")
    return {"headers": d.get("headers", []), "rows": d.get("rows", [])}


class IdReq(BaseModel):
    id: str = Field(min_length=10, max_length=80)


@app.post("/api/admin/unhide")
def admin_unhide(request: Request, body: IdReq, x_admin_token: str = Header("")):
    a = admin_auth(request, x_admin_token)
    with GONE_LOCK:
        GONE_AUTO.discard(body.id); GONE_MANUAL.discard(body.id)
        refresh_mask()
    save_gone_file()
    still = body.id in GONE_REMOTE
    audit(a["name"], a["ip"], "คืนรูปให้ค้นเจอ", f"รหัส {body.id} · {drive_link(body.id)}" + (" (ยังถูกซ่อนจากสถานะในชีต)" if still else ""), not still)
    return {"ok": not still, "message": "รูปนี้ยังถูกซ่อนจากสถานะ \"อนุมัติ\" ในชีต — ต้องเปลี่ยนสถานะในชีตก่อน" if still else "คืนรูปแล้ว"}


@app.get("/api/admin/requests")
def admin_requests(request: Request, x_admin_token: str = Header("")):
    a = admin_auth(request, x_admin_token)
    try:
        rows = sheet_call({"action": "requests"}).get("rows", [])
    except HTTPException as e:
        audit(a["name"], a["ip"], "ดูคำขอลบรูป", str(e.detail), False)
        raise
    gone = gone_ids()
    out = []
    for r in rows[:300]:
        m = DRIVE_ID.search(str(r.get("photo_id") or r.get("photo_url") or ""))
        pid = m.group(0) if m else ""
        out.append({**{k: r.get(k, "") for k in ("ref", "time", "name", "grade", "student_id", "reason", "face_match", "status")},
                    **photo_row(pid), "hidden": pid in gone} if pid else {**r, "id": "", "thumb": "", "open": "", "hidden": False, "in_index": False})
    audit(a["name"], a["ip"], "ดูคำขอลบรูป", f"{len(out)} รายการ")
    return {"rows": out}


class DecideReq(BaseModel):
    ref: str = Field(pattern=r"^DEL-[0-9]{6}-[0-9A-Fa-f]{4}$")
    decision: Literal["approve", "reject"]


@app.post("/api/admin/decide")
def admin_decide(request: Request, body: DecideReq, x_admin_token: str = Header("")):
    """อนุมัติ = เปลี่ยนสถานะในชีต + ให้ Apps Script ย้ายไฟล์ Drive เข้าถังขยะ + ซ่อนจากผลค้นหาทันที / ปฏิเสธ = เปลี่ยนสถานะอย่างเดียว"""
    a = admin_auth(request, x_admin_token)
    word = "อนุมัติ&ลบรูป" if body.decision == "approve" else "ปฏิเสธคำขอลบรูป"
    try:
        d = sheet_call({"action": "set_status", "ref": body.ref, "decision": body.decision})
    except HTTPException as e:
        audit(a["name"], a["ip"], word, f"{body.ref} — {e.detail}", False)
        raise
    m = DRIVE_ID.search(str(d.get("photo_id") or ""))
    pid = m.group(0) if m else ""
    hidden = False
    if body.decision == "approve" and pid in S["ix"]["index"]:
        with GONE_LOCK:
            GONE_MANUAL.add(pid)
            refresh_mask()
        save_gone_file()
        hidden = True
    audit(a["name"], a["ip"], word, f"{body.ref} · รหัสรูป {pid} · {drive_link(pid) if pid else '-'} · ไฟล์ Drive: {'ย้ายเข้าถังขยะแล้ว' if d.get('trashed') else 'ไม่ได้ลบ'}",
          body.decision == "reject" or bool(d.get("trashed")))
    return {"ok": True, "status": d.get("status"), "trashed": d.get("trashed"), "hidden": hidden, "photo_id": pid, "message": d.get("message", "")}


@app.post("/api/admin/reload")
def admin_reload(request: Request, x_admin_token: str = Header("")):
    """เพิ่มรูปเข้าฐานข้อมูลแล้วไม่ต้อง redeploy: อัปไฟล์ embeddings.npz ใหม่ไปที่ EMB_URL แล้วเรียกอันนี้ (หรือรอรีสตาร์ท)"""
    a = admin_auth(request, x_admin_token)
    try:
        ok = load_index(force_download=True)
    except Exception as e:
        audit(a["name"], a["ip"], "โหลดฐานข้อมูลใบหน้าใหม่", f"ล้มเหลว: {e}", False)
        raise HTTPException(502, f"โหลดฐานข้อมูลใหม่ไม่สำเร็จ (ยังใช้ชุดเดิมอยู่): {e}")
    fetch_removed()
    load_admins()
    ix = S["ix"]
    audit(a["name"], a["ip"], "โหลดฐานข้อมูลใบหน้าใหม่", f"{len(ix['names'])} รูป / {len(ix['emb'])} ใบหน้า / แอดมินที่ลงทะเบียน {len(ADMINS_STATE['list'])} คน")
    return {"success": ok, "photos": int(len(ix["names"])), "faces": int(len(ix["emb"])), "hidden": int(ix["hide"].sum())}


@app.get("/api/admin/auditlog")
def admin_auditlog(request: Request, x_admin_token: str = Header("")):
    """บันทึกการใช้งานแอดมินจากชีต (แท็บ AdminLog) — ถ้าชีตอ่านไม่ได้ จะแสดงที่เก็บในหน่วยความจำของเซิร์ฟเวอร์แทน"""
    a = admin_auth(request, x_admin_token)
    audit(a["name"], a["ip"], "ดูบันทึกการใช้งานแอดมิน")
    try:
        d = sheet_call({"action": "adminlog"})
        return {"headers": d.get("headers", []), "rows": d.get("rows", []), "source": "sheet"}
    except HTTPException as e:
        return {"headers": ["เวลา", "ผู้ดูแล", "IP", "การกระทำ", "รายละเอียด", "ผล"],
                "rows": [[r["time"], r["admin"], r["ip"], r["action"], r["detail"], r["result"]] for r in AUDIT_BUF], "source": "memory", "note": str(e.detail)}


@app.post("/api/delete/verify")
def delete_verify(request: Request, file: UploadFile = File(...), photo_id: str = Form(...)):
    ip = client_ip(request)
    limit(dhits, ip, DELETE_RATE, 600, "ตรวจใบหน้าถี่เกินไป รอสักครู่แล้วลองใหม่")
    need_ready()
    ix = S["ix"]
    i = ix["index"].get(photo_id)
    if i is None or ix["hide"][i]:
        raise HTTPException(404, "ไม่พบรูปนี้ในระบบ")
    faces = detect(read_image(file))
    if not faces:
        return {"success": False, "message": "ไม่พบใบหน้าในรูปที่สแกน ลองอยู่ในที่สว่าง มองกล้องตรง ๆ"}
    me = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    rows = np.where(ix["owner"] == i)[0]
    sim = float(score(ix, me.normed_embedding.astype(np.float32), rows).max()) if len(rows) else -1.0
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
    limit(dhits, ip, DELETE_RATE, 600, "ส่งคำขอถี่เกินไป รอสักครู่แล้วลองใหม่")
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
    nm = re.sub(r"[<>@`]", "", row["name"])
    notify(f"🗑️ คำขอลบรูปใหม่ {ref}\nผู้ขอ: {nm} ({re.sub(r'[<>@`]', '', row['grade'])}) · ใบหน้าตรง {row['face_match']}%\n"
           f"รหัสรูป: {pid}\nลิงก์ Drive: {row['photo_url']}\nไปอนุมัติ/ปฏิเสธได้ที่หน้าแอดมิน")
    return {"success": True, "ref": ref}


for d in ("photos", "thumbs"):
    os.makedirs(f"data/{d}", exist_ok=True)
    app.mount(f"/{d}", StaticFiles(directory=f"data/{d}"), name=d)


@app.get("/")
def home():
    return FileResponse("index.html", headers={"Cache-Control": "no-cache"})
