#!/usr/bin/env python3
"""
จัดการฐานข้อมูลใบหน้า (data/embeddings.npz) แบบ "เพิ่มเฉพาะรูปใหม่ / ตัดรูปที่ถูกลบ" ไม่ต้องสร้างใหม่ทั้งชุด

  python build_index.py info   --emb data/embeddings.npz
  python build_index.py sync   --folder <DRIVE_FOLDER_ID> --api-key <KEY>          # เทียบโฟลเดอร์ Drive กับฐานข้อมูล: เพิ่มรูปใหม่ + ตัดรูปที่หายไป
  python build_index.py sync   --folder <ID> --api-key <KEY> --exclude-file removed.txt   # รูปที่แอดมินอนุมัติลบ (บรรทัดละ id/ลิงก์) จะถูกตัดออกด้วย
  python build_index.py remove --ids ID1 ID2 ...  |  --file removed.txt             # ตัดรูปออกจากฐานข้อมูลเฉยๆ ไม่ต้องต่อ Drive
  python build_index.py sync   ... --dry-run                                        # ดูก่อนว่าจะเพิ่ม/ตัดกี่รูป ยังไม่แตะไฟล์

หมายเหตุ
 - รองรับโฟลเดอร์ย่อย; ต้องแชร์โฟลเดอร์แบบ "ทุกคนที่มีลิงก์" ถ้าใช้ --api-key (ถ้าไม่ได้แชร์สาธารณะ ใช้ --token <OAuth access token> แทน)
 - รูปที่ไม่มีใบหน้าก็ถูกบันทึกชื่อไว้ (ไม่มีแถว embedding) เพื่อไม่ต้องสแกนซ้ำทุกครั้ง — เหมือนไฟล์เดิมของคุณ
 - บันทึกไฟล์แบบอะตอมมิก + checkpoint ทุก N รูป ถ้าหยุดกลางทางรันซ้ำแล้วทำต่อจากเดิมได้
 - ก่อนเขียนทับไฟล์ครั้งแรกของการรัน จะสำรองเป็น embeddings.npz.bak-วันเวลา อัตโนมัติ (เก็บ 3 ชุดล่าสุด)
 - --limit N = โหมดทดสอบ: เพิ่มแค่ N รูปและ "ไม่ตัดรูปใดออก"
 - การตัดรูปออกจะลบ embedding (ข้อมูลชีวมิติ) ของรูปนั้นทิ้งจริง ไม่ใช่แค่ซ่อน
"""
import argparse, io, json, os, re, sys, time, urllib.error, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

SCALE = 255.0                      # ไฟล์เดิม: int8 ของ embedding ที่ normalize แล้ว × 255 (ตรวจจากค่า norm ≈ 255)
DRIVE = "https://www.googleapis.com/drive/v3/files"
FOLDER = "application/vnd.google-apps.folder"
DRIVE_ID = re.compile(r"[A-Za-z0-9_-]{20,}")


# ───────────────────────── ไฟล์ฐานข้อมูล ─────────────────────────
def load_db(path, model):
    if not os.path.exists(path):
        return {"emb": np.zeros((0, 512), np.int8), "owner": np.zeros(0, np.int32), "names": np.array([], dtype=str), "model": model}
    z = np.load(path)
    emb = z["emb"] if "emb" in z else z["embeddings"]
    m = str(z["model"]) if "model" in z else model
    if m != model:
        sys.exit(f"❌ ฐานข้อมูลสร้างจากโมเดล {m} แต่สั่งใช้ {model} — embedding ต่างโมเดลกันเทียบกันไม่ได้ (ใช้ --model {m})")
    owner = z["owner"] if "owner" in z else np.arange(len(emb), dtype=np.int32)
    return {"emb": emb, "owner": owner.astype(np.int32), "names": z["names"], "model": m}


_BACKED = set()


def backup_once(path, keep=3):
    """ก่อนเขียนทับไฟล์เดิมครั้งแรกของการรัน สำเนาเป็น embeddings.npz.bak-วันเวลา (เก็บ 3 ชุดล่าสุด) — ไม่ต้องจำว่าต้องสำรองเอง"""
    if path in _BACKED or not os.path.exists(path):
        return
    _BACKED.add(path)
    dst = f"{path}.bak-{time.strftime('%Y%m%d-%H%M%S')}"
    import shutil
    shutil.copy2(path, dst)
    olds = sorted(f for f in os.listdir(os.path.dirname(os.path.abspath(path))) if f.startswith(os.path.basename(path) + ".bak-"))
    for f in olds[:-keep]:
        os.remove(os.path.join(os.path.dirname(os.path.abspath(path)), f))
    print(f"สำรองไฟล์เดิมไว้ที่ {dst}", flush=True)


def save_db(path, db):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    backup_once(path)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, emb=db["emb"], owner=db["owner"], names=np.array(db["names"], dtype=str),
                            model=np.array(db["model"]), source=np.array("drive"))
    with np.load(tmp) as z:                                        # อ่านกลับมาตรวจก่อนสลับไฟล์จริง
        assert len(z["emb"]) == len(z["owner"]) and (len(z["owner"]) == 0 or z["owner"].max() < len(z["names"]))
    os.replace(tmp, path)


def quantize(normed):
    return np.clip(np.rint(np.asarray(normed, np.float32) * SCALE), -127, 127).astype(np.int8)


def purge(db, remove):
    """ตัดรูป (และใบหน้าทั้งหมดของรูปนั้น) ออก แล้วรันเลข owner ใหม่ให้ต่อเนื่อง"""
    names = np.asarray(db["names"])
    keep = np.array([str(n) not in remove for n in names], bool)
    if keep.all():
        return db, 0, 0
    new_idx = np.cumsum(keep) - 1
    rows = keep[db["owner"]]
    out = {"emb": db["emb"][rows], "owner": new_idx[db["owner"][rows]].astype(np.int32), "names": names[keep], "model": db["model"]}
    return out, int((~keep).sum()), int((~rows).sum())


def append(db, new_names, faces_per_photo):
    base = len(db["names"])
    rows, owners = [], []
    for j, faces in enumerate(faces_per_photo):
        for f in faces:
            rows.append(quantize(f)); owners.append(base + j)
    emb = np.concatenate([db["emb"], np.stack(rows)]) if rows else db["emb"]
    owner = np.concatenate([db["owner"], np.array(owners, np.int32)]) if owners else db["owner"]
    return {"emb": emb, "owner": owner, "names": np.concatenate([np.asarray(db["names"], dtype=str), np.array(new_names, dtype=str)]) if len(new_names) else db["names"], "model": db["model"]}


def ids_from(args_ids, file):
    raw = " ".join(args_ids or [])
    if file:
        with open(file, encoding="utf-8") as f:
            raw += " " + f.read()
    return set(DRIVE_ID.findall(raw))


# ───────────────────────── Google Drive ─────────────────────────
def http(url, params, key, token, tries=5):
    q = {k: v for k, v in params.items() if v is not None}
    if key:
        q["key"] = key
    req = urllib.request.Request(url + "?" + urllib.parse.urlencode(q), headers={"Authorization": f"Bearer {token}"} if token else {})
    for a in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and a < tries - 1:
                time.sleep(2 ** a); continue
            raise
        except (urllib.error.URLError, TimeoutError):
            if a < tries - 1:
                time.sleep(2 ** a); continue
            raise


def list_images(folder, key, token):
    """คืน {id: name} ของรูปทั้งหมดในโฟลเดอร์และโฟลเดอร์ย่อย (ไม่รวมที่อยู่ในถังขยะ)"""
    found, stack = {}, [folder]
    while stack:
        cur, page = stack.pop(), None
        while True:
            data = json.loads(http(DRIVE, {"q": f"'{cur}' in parents and trashed=false", "fields": "nextPageToken,files(id,name,mimeType)", "pageSize": 1000,
                                           "pageToken": page, "supportsAllDrives": "true", "includeItemsFromAllDrives": "true"}, key, token))
            for f in data.get("files", []):
                if f["mimeType"] == FOLDER:
                    stack.append(f["id"])
                elif f["mimeType"].startswith("image/"):
                    found[f["id"]] = f["name"]
            page = data.get("nextPageToken")
            if not page:
                break
    return found


def download(pid, key, token):
    return http(f"{DRIVE}/{pid}", {"alt": "media", "supportsAllDrives": "true"}, key, token)


# ───────────────────────── ตรวจจับใบหน้า ─────────────────────────
def make_detector(model, det_size, max_side):
    import cv2
    from insightface.app import FaceAnalysis
    from PIL import Image, ImageOps
    fa = FaceAnalysis(name=model, allowed_modules=["detection", "recognition"], providers=["CPUExecutionProvider"])
    fa.prepare(ctx_id=0, det_size=(det_size, det_size))

    def faces_of(data):
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
        im.thumbnail((max_side, max_side))
        return [f.normed_embedding for f in fa.get(cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR))]
    return faces_of


# ───────────────────────── คำสั่ง ─────────────────────────
def cmd_info(a):
    db = load_db(a.emb, a.model)
    n, f = len(db["names"]), len(db["emb"])
    with_face = len(np.unique(db["owner"]))
    print(f"รูปทั้งหมด {n:,} | มีใบหน้า (ค้นเจอได้) {with_face:,} | ไม่พบใบหน้าตอนสแกน {n - with_face:,} | ใบหน้ารวม {f:,} | โมเดล {db['model']}")


def cmd_remove(a):
    ids = ids_from(a.ids, a.file)
    db = load_db(a.emb, a.model)
    db, p, f = purge(db, ids)
    print(f"ตัดออก {p} รูป ({f} ใบหน้า) จากที่ขอ {len(ids)} id")
    if p and not a.dry_run:
        save_db(a.emb, db); print(f"บันทึก {a.emb} แล้ว (เหลือ {len(db['names']):,} รูป)")


def cmd_sync(a, lister=list_images, fetch=download, detector=None):
    db = load_db(a.emb, a.model)
    known = {str(n) for n in db["names"]}
    exclude = ids_from(None, a.exclude_file) if a.exclude_file else set()
    print("กำลังอ่านรายการรูปจาก Drive …", flush=True)
    current = lister(a.folder, a.api_key, a.token)
    if not current:
        sys.exit("❌ Drive ไม่คืนรูปเลย — เช็ค folder id / สิทธิ์แชร์ / api key (ยกเลิกเพื่อไม่ให้ฐานข้อมูลถูกล้าง)")
    cur_ids = set(current) - exclude
    new = sorted(cur_ids - known)
    gone = (known - cur_ids)
    pct = 100 * len(gone) / max(1, len(known))
    print(f"ใน Drive {len(current):,} รูป | ในฐานข้อมูล {len(known):,} | รูปใหม่ {len(new):,} | จะตัดออก {len(gone):,} ({pct:.1f}%)")
    if a.dry_run:
        return
    if gone and pct > a.max_purge_pct and not a.force:
        sys.exit(f"❌ จะตัดรูปออกมากกว่า {a.max_purge_pct}% — น่าจะระบุโฟลเดอร์ผิดหรือ Drive ตอบไม่ครบ ถ้าแน่ใจให้เพิ่ม --force")
    if a.limit and gone:                                         # โหมดทดสอบ (--limit) ต้องไม่ลบอะไรเลย — เวอร์ชันก่อนหน้าลบก่อนแล้วค่อยเช็ก --limit
        print(f"โหมดทดสอบ --limit: ข้ามการตัดรูป {len(gone)} รูป (จะตัดเมื่อรันโดยไม่ใส่ --limit)")
        gone = set()
    if gone:
        db, p, f = purge(db, gone)
        print(f"ตัดออกแล้ว {p} รูป ({f} ใบหน้า)")
        save_db(a.emb, db)
    if a.limit:
        new = new[:a.limit]
    if not new:
        print("ไม่มีรูปใหม่ ✔"); return
    detect = detector or make_detector(a.model, a.det_size, a.max_side)
    t0, added_faces, no_face, failed, done = time.time(), 0, 0, 0, 0
    batch_names, batch_faces = [], []

    def flush():
        nonlocal db, batch_names, batch_faces
        if batch_names:
            db = append(db, batch_names, batch_faces)
            save_db(a.emb, db)
            batch_names, batch_faces = [], []

    with ThreadPoolExecutor(a.workers) as ex:
        for s in range(0, len(new), 32):                         # ดาวน์โหลดล่วงหน้าทีละ 32 รูป ไม่กินแรมเกิน
            chunk = new[s:s + 32]
            def grab(pid):
                try:
                    return fetch(pid, a.api_key, a.token)
                except Exception as e:
                    return e
            for pid, data in zip(chunk, ex.map(grab, chunk)):
                done += 1
                if isinstance(data, Exception):
                    failed += 1; print(f"  ⚠️ ดาวน์โหลด {pid} ไม่สำเร็จ ({data}) — ข้าม รอบหน้าจะลองใหม่", flush=True); continue
                try:
                    faces = detect(data)
                except Exception as e:
                    faces = []; print(f"  ⚠️ อ่านรูป {pid} ไม่ได้ ({e}) — บันทึกเป็นรูปไม่มีใบหน้า", flush=True)
                batch_names.append(pid); batch_faces.append(faces)
                added_faces += len(faces); no_face += not faces
                if len(batch_names) >= a.checkpoint:
                    flush()
                if done % 25 == 0:
                    rate = done / (time.time() - t0)
                    print(f"  {done}/{len(new)} รูป | {rate:.1f} รูป/วินาที | เหลือราว {int((len(new) - done) / max(rate, 1e-6) / 60)} นาที", flush=True)
    flush()
    print(f"เสร็จ: เพิ่ม {len(new) - failed:,} รูป ({added_faces:,} ใบหน้า, ไม่พบใบหน้า {no_face:,}) ดาวน์โหลดพลาด {failed} | ฐานข้อมูลตอนนี้ {len(db['names']):,} รูป")
    print("ขั้นต่อไป: อัปโหลด", a.emb, "ไปที่ EMB_URL แล้วกดปุ่ม “โหลดฐานข้อมูลใบหน้าใหม่” ในหน้าแอดมิน")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("info", "remove", "sync"):
        s = sub.add_parser(name)
        s.add_argument("--emb", default="data/embeddings.npz")
        s.add_argument("--model", default=os.getenv("MODEL_NAME", "buffalo_sc"))
        if name != "info":
            s.add_argument("--dry-run", action="store_true")
        if name == "remove":
            s.add_argument("--ids", nargs="*"); s.add_argument("--file")
        if name == "sync":
            s.add_argument("--folder", required=True)
            s.add_argument("--api-key", default=os.getenv("GOOGLE_API_KEY"))
            s.add_argument("--token", default=os.getenv("GOOGLE_ACCESS_TOKEN"))
            s.add_argument("--exclude-file")
            s.add_argument("--det-size", type=int, default=640)
            s.add_argument("--max-side", type=int, default=1600)
            s.add_argument("--workers", type=int, default=4)
            s.add_argument("--checkpoint", type=int, default=300)
            s.add_argument("--limit", type=int, default=0, help="โหมดทดสอบ: เพิ่มแค่ N รูปแรก และไม่ตัดรูปใดออก")
            s.add_argument("--max-purge-pct", type=float, default=20)
            s.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    if a.cmd == "sync" and not (a.api_key or a.token):
        sys.exit("ต้องระบุ --api-key (หรือ --token) สำหรับอ่าน Drive")
    {"info": cmd_info, "remove": cmd_remove, "sync": cmd_sync}[a.cmd](a)


if __name__ == "__main__":
    main()
