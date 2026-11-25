#!/usr/bin/env python3
"""
ลงทะเบียน "ใบหน้าแอดมิน" สำหรับล็อกอินหน้าแอดมิน (ขั้นที่ 2 หลังกรอกรหัสผ่าน) และย้อนดูว่ารหัสใบหน้าแต่ละชุดเป็นของใคร

  python enroll_admin.py add  --name "ครูสมชาย ใจดี" --role "ผู้ดูแลหลัก" --photo a.jpg --photo b.jpg --photo c.jpg
  python enroll_admin.py list                        # ดูว่ามีใครลงทะเบียนอยู่บ้าง (id / ชื่อ / ตำแหน่ง / วันที่ / จำนวนรหัสใบหน้า)
  python enroll_admin.py who  --photo ไม่รู้ว่าใคร.jpg   # เอารูปมาเทียบ: ใบหน้านี้ตรงกับแอดมินคนไหน กี่ %
  python enroll_admin.py remove --id a1              # (หรือ --name "ชื่อ")
  python enroll_admin.py export-env                  # พิมพ์ JSON บรรทัดเดียว ไว้วางใน env ADMIN_FACES บน Render (ถ้าไม่อยากเอาไฟล์ขึ้น GitHub)

คำแนะนำ
 - ใช้รูปของคนเดียวกัน 3–5 รูป คนละมุม/แสง (หน้าตรง, เอียงเล็กน้อย, มีแว่น/ไม่มีแว่น) แต่ละรูปต้องเห็นหน้าชัด จะสแกนผ่านง่ายและแม่นขึ้น
 - ไฟล์ admins.json = ข้อมูลชีวมิติ ห้ามเอาขึ้น GitHub แบบสาธารณะ (ใช้ repo แบบ Private หรือใช้ env ADMIN_FACES แทน)
 - ชื่อที่ใส่ใน --name คือชื่อที่หน้าแอดมินแสดง และชื่อที่ถูกบันทึกลงชีตทุกครั้งที่ผู้นั้นทำอะไรในระบบ
 - ต้องติดตั้ง insightface + onnxruntime (ชุดเดียวกับ requirements.txt) — เปิดใน GitHub Codespaces แล้ว pip install -r requirements.txt ได้เลย
"""
import argparse, json, os, re, sys, time
from datetime import datetime, timedelta, timezone

import numpy as np

SCALE = 255.0                       # เหมือน embeddings.npz: เวกเตอร์ที่ normalize แล้ว × 255 เก็บเป็น int8
TH = timezone(timedelta(hours=7))
README = ("ใบหน้าแอดมิน — สร้าง/แก้ด้วย enroll_admin.py เท่านั้น (อย่าแก้ตัวเลขด้วยมือ) | name = ชื่อที่หน้าแอดมินแสดงและที่บันทึกลงชีต | "
          "embeddings = รหัสใบหน้า (เวกเตอร์ 512 ค่า) 1 รูป = 1 บรรทัด | ข้อมูลนี้คือข้อมูลชีวมิติ ห้ามเผยแพร่")


def load(path):
    if not os.path.exists(path):
        return {"admins": []}
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    d.setdefault("admins", [])
    return d


def dumps(data):
    """เขียน JSON ให้อ่านง่าย: ข้อมูลแอดมินแต่ละคนอยู่ไม่กี่บรรทัด รหัสใบหน้า 1 รูป = 1 บรรทัด (json.dumps indent จะแตกเป็นเป็นพันบรรทัด)"""
    out = ['{', f'  "_readme": {json.dumps(README, ensure_ascii=False)},', '  "admins": [']
    for i, a in enumerate(data["admins"]):
        meta = {k: a[k] for k in a if k != "embeddings"}
        out.append('    {' + ", ".join(f'{json.dumps(k)}: {json.dumps(v, ensure_ascii=False)}' for k, v in meta.items()) + ',')
        out.append('     "embeddings": [')
        out.append(",\n".join("       " + json.dumps(e, separators=(",", ":")) for e in a["embeddings"]))
        out.append('     ]}' + ("," if i < len(data["admins"]) - 1 else ""))
    out += ['  ]', '}']
    return "\n".join(out) + "\n"


def save(path, data):
    text = dumps(data)
    json.loads(text)                                           # ตรวจว่าเป็น JSON ที่ถูกต้องก่อนเขียนทับ
    if os.path.exists(path):
        import shutil
        shutil.copy2(path, f"{path}.bak-{time.strftime('%Y%m%d-%H%M%S')}")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def quantize(v):
    return np.clip(np.rint(np.asarray(v, np.float32) * SCALE), -127, 127).astype(int).tolist()


def unit(e):
    e = np.asarray(e, np.float32)
    return e / np.linalg.norm(e, axis=-1, keepdims=True)


def make_detector(model="buffalo_sc", det_size=640):
    import cv2
    from insightface.app import FaceAnalysis
    from PIL import Image, ImageOps
    fa = FaceAnalysis(name=model, allowed_modules=["detection", "recognition"], providers=["CPUExecutionProvider"])
    fa.prepare(ctx_id=0, det_size=(det_size, det_size))

    def faces_of(path):
        im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        im.thumbnail((1280, 1280))
        fs = fa.get(cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR))
        return sorted(((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), f.normed_embedding) for f in fs)[::-1]   # ใหญ่สุดก่อน
    return faces_of


def one_face(detect, path):
    fs = detect(path)
    if not fs:
        sys.exit(f"❌ ไม่พบใบหน้าในรูป {path} — ใช้รูปที่เห็นหน้าชัด สว่าง ไม่ถูกบัง")
    if len(fs) > 1:
        print(f"  ⚠️ {path}: พบ {len(fs)} ใบหน้า ใช้ใบหน้าที่ใหญ่ที่สุด (ถ้าไม่ใช่คนที่ต้องการ ให้ครอปรูปก่อน)")
    return np.asarray(fs[0][1], np.float32)


def cmd_add(a, detect=None):
    data = load(a.file)
    detect = detect or make_detector()
    embs = [one_face(detect, p) for p in a.photo]
    if len(embs) > 1:
        m = unit(embs) @ unit(embs).T
        low = float(m[np.triu_indices(len(embs), 1)].min())
        print(f"  ความเหมือนระหว่างรูปที่ใช้ลงทะเบียน (ต่ำสุด) = {low * 100:.0f}%")
        if low < 0.3:
            sys.exit("❌ รูปที่ให้มาดูเหมือนเป็นคนละคนกัน (ความเหมือนต่ำมาก) — ตรวจรูปอีกครั้ง ยังไม่ได้บันทึกอะไร")
    same = [x for x in data["admins"] if x["name"] == a.name or (a.id and x["id"] == a.id)]
    if same and not a.replace:
        sys.exit(f"❌ มี \"{a.name}\" อยู่แล้ว (id {same[0]['id']}) — ใช้ --replace เพื่อแทนที่รหัสใบหน้าเดิม")
    pos = len(data["admins"])
    if same:                                                   # แทนที่ตำแหน่งเดิม (ลำดับในไฟล์ไม่เปลี่ยน)
        pos = data["admins"].index(same[0])
        data["admins"].remove(same[0])
    nid = a.id or (same[0]["id"] if same else "a" + str(1 + max([int(re.sub(r"\D", "", x["id"]) or 0) for x in data["admins"]] + [0])))
    data["admins"].insert(pos, {"id": nid, "name": a.name, "role": a.role if a.role is not None else (same[0].get("role", "") if same else ""), "enrolled": datetime.now(TH).strftime("%Y-%m-%d"), "note": a.note if a.note is not None else (same[0].get("note", "") if same else ""),
                           "photos": len(embs), "embeddings": [quantize(e) for e in embs]})
    save(a.file, data)
    print(f"✅ ลงทะเบียน \"{a.name}\" (id {nid}) ด้วย {len(embs)} รูป → {a.file}  (ตอนนี้มี {len(data['admins'])} คน)")


def cmd_list(a):
    ad = load(a.file)["admins"]
    if not ad:
        print("ยังไม่มีแอดมินที่ลงทะเบียนใบหน้า (หน้าแอดมินจะเข้าด้วยรหัสผ่านอย่างเดียว)"); return
    print(f"{'id':<6}{'ชื่อ':<28}{'ตำแหน่ง':<18}{'ลงทะเบียน':<12}{'รหัสใบหน้า':<10} หมายเหตุ")
    for x in ad:
        print(f"{x['id']:<6}{x['name']:<28}{x.get('role', ''):<18}{x.get('enrolled', ''):<12}{len(x['embeddings']):<10} {x.get('note', '')}")


def cmd_who(a, detect=None):
    ad = load(a.file)["admins"]
    if not ad:
        sys.exit("ยังไม่มีแอดมินที่ลงทะเบียน")
    detect = detect or make_detector()
    q = unit(one_face(detect, a.photo))
    res = sorted(((float((unit(x["embeddings"]) @ q).max()), x) for x in ad), key=lambda t: -t[0])
    for s, x in res:
        print(f"  {s * 100:5.1f}%  {x['name']} (id {x['id']}{', ' + x['role'] if x.get('role') else ''})")
    s, x = res[0]
    print(f"→ ตรงกับ \"{x['name']}\"" if s >= a.threshold else f"→ ไม่ตรงกับใครเลย (สูงสุด {s * 100:.1f}% < เกณฑ์ {a.threshold * 100:.0f}%)")


def cmd_remove(a):
    data = load(a.file)
    hit = [x for x in data["admins"] if (a.id and x["id"] == a.id) or (a.name and x["name"] == a.name)]
    if not hit:
        sys.exit("❌ ไม่พบแอดมินที่ระบุ (ดู id จาก: python enroll_admin.py list)")
    for x in hit:
        data["admins"].remove(x)
    save(a.file, data)
    print(f"ลบ \"{hit[0]['name']}\" แล้ว (เหลือ {len(data['admins'])} คน)" + (" — ตอนนี้ไม่มีใบหน้าแอดมินเลย ระบบจะกลับไปใช้รหัสผ่านอย่างเดียว" if not data["admins"] else ""))


def cmd_export_env(a):
    d = load(a.file)
    print(json.dumps({"admins": d["admins"]}, ensure_ascii=False, separators=(",", ":")))


def main(argv=None, detect=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for n in ("add", "list", "who", "remove", "export-env"):
        s = sub.add_parser(n)
        s.add_argument("--file", default="admins.json")
        if n == "add":
            s.add_argument("--name", required=True); s.add_argument("--role"); s.add_argument("--note"); s.add_argument("--id")
            s.add_argument("--photo", action="append", required=True); s.add_argument("--replace", action="store_true")
        if n == "who":
            s.add_argument("--photo", required=True); s.add_argument("--threshold", type=float, default=0.5)
        if n == "remove":
            s.add_argument("--id"); s.add_argument("--name")
    a = p.parse_args(argv)
    {"add": lambda: cmd_add(a, detect), "list": lambda: cmd_list(a), "who": lambda: cmd_who(a, detect), "remove": lambda: cmd_remove(a), "export-env": lambda: cmd_export_env(a)}[a.cmd]()


if __name__ == "__main__":
    main()
