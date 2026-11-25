/**
 * Apps Script ฝั่งชีต สำหรับระบบ STSS (ไฟล์เดียว รวมของเดิมที่ชื่อ apps_script_removed.gs — ใช้ไฟล์นี้แทนได้เลย)
 * "ยังไม่ได้ทดสอบกับชีตจริง" — ผมไม่เห็นสคริปต์ doPost เดิมและหัวคอลัมน์ของคุณ ให้ตรวจค่าคงที่ด้านล่างให้ตรงก่อนใช้
 *
 * ติดตั้ง (รายละเอียดเต็มดู README.md ข้อ 6)
 *  1) วางโค้ดนี้ในโปรเจกต์ Apps Script เดิม (ถ้ามี doGet อยู่แล้วให้รวมกัน)
 *  2) เพิ่ม 2 บรรทัดนี้ "บนสุด" ของ doPost เดิม (ไม่ต้องแก้อย่างอื่นใน doPost):
 *        var _d = {}; try { _d = JSON.parse(e.postData.contents); } catch (x) {}
 *        if (_d.type === 'admin_log') return adminLog_(_d);
 *  3) Project Settings → Script properties → เพิ่ม  ADMIN_KEY = รหัสเดียวกับ ADMIN_KEY ที่ตั้งบน Render
 *  4) Deploy → Manage deployments → แก้ไข Web app → Execute as: Me / Who has access: Anyone → New version → Deploy
 *  5) รันฟังก์ชัน authorize_ จากตัวแก้ไข 1 ครั้งเพื่ออนุญาตสิทธิ์ Drive
 *
 * GET:  removed (สาธารณะ) | requests, history, adminlog, set_status, trash (ต้องมี key=ADMIN_KEY)
 * POST: type=admin_log (ต้องมี key=ADMIN_KEY ในเนื้อข้อความ) → เขียนบันทึกการใช้งานแอดมินลงแท็บ AdminLog (สร้างให้เองถ้ายังไม่มี)
 * ไฟล์ที่จะลบต้องเป็นไฟล์ที่เจ้าของสคริปต์เป็นเจ้าของหรือมีสิทธิ์ลบ; setTrashed = ย้ายเข้าถังขยะ (กู้คืนได้ 30 วัน) ไม่ใช่ลบถาวร
 */

// ── 1) ชื่อแท็บในชีต ──
const REQUEST_SHEET = 'Sheet1';      // แท็บที่เก็บแถว delete_request
const SEARCH_SHEET = 'Sheet2';       // แท็บที่เก็บแถวประวัติการค้นหา (type = search)
const ADMIN_LOG_SHEET = 'AdminLog';  // แท็บบันทึกการใช้งานแอดมิน (สร้างอัตโนมัติ)
// ── 2) สถานะ ──
const ST_PENDING = 'รอพิจารณา', ST_APPROVED = 'อนุมัติ', ST_DELETED = 'ลบแล้ว', ST_REJECTED = 'ปฏิเสธ';
const HIDE_STATUSES = [ST_APPROVED, ST_DELETED];   // สถานะที่ถือว่า "ต้องซ่อนรูปนี้" (ขึ้นต้นด้วยคำนี้ก็นับ)
// ── 3) ชื่อหัวคอลัมน์ที่เป็นไปได้ (ไม่สนตัวพิมพ์) ──
const COLS = {
  ref: ['ref', 'รหัสอ้างอิง'], time: ['time', 'เวลา'], photo_id: ['photo_id', 'photo id', 'รหัสรูป'],
  name: ['name', 'ชื่อ', 'ชื่อ-นามสกุล'], grade: ['grade', 'ชั้น'], student_id: ['student_id', 'รหัสประจำตัว'],
  reason: ['reason', 'เหตุผล'], face_match: ['face_match', 'ความตรงของใบหน้า'], status: ['status', 'สถานะ']
};

function json_(o) { return ContentService.createTextOutput(JSON.stringify(o)).setMimeType(ContentService.MimeType.JSON); }
function sheet_(name) {
  const s = SpreadsheetApp.getActive().getSheetByName(name);
  if (!s) throw new Error('ไม่พบแท็บ "' + name + '" — แก้ค่า REQUEST_SHEET / SEARCH_SHEET ด้านบนสคริปต์');
  return s;
}
function colIndex_(headers, key) {
  const names = COLS[key].map(function (x) { return x.toLowerCase(); });
  for (var i = 0; i < headers.length; i++) if (names.indexOf(String(headers[i]).trim().toLowerCase()) >= 0) return i;
  throw new Error('ไม่พบคอลัมน์ "' + COLS[key][0] + '" ในแท็บ ' + REQUEST_SHEET + ' (หัวตารางที่มี: ' + headers.join(', ') + ')');
}
function cell_(v) { return v instanceof Date ? Utilities.formatDate(v, 'Asia/Bangkok', 'yyyy-MM-dd HH:mm:ss') : (v === null || v === undefined ? '' : String(v)); }
function idOf_(v) { var m = String(v).match(/[A-Za-z0-9_-]{20,}/); return m ? m[0] : ''; }
function authed_(e) {
  var k = PropertiesService.getScriptProperties().getProperty('ADMIN_KEY');
  return !!k && e.parameter && e.parameter.key === k;
}

function doGet(e) {
  try {
    var action = (e && e.parameter && e.parameter.action) || '';
    if (action === 'removed') return removed_();
    if (!action) return json_({ ok: true, service: 'stss' });
    if (!authed_(e)) return json_({ ok: false, error: 'key ไม่ถูกต้อง หรือยังไม่ได้ตั้ง ADMIN_KEY ใน Script properties' });
    if (action === 'requests') return requests_();
    if (action === 'history') return history_(parseInt(e.parameter.limit, 10) || 500);
    if (action === 'adminlog') return adminLogRead_();
    if (action === 'set_status') return setStatus_(e.parameter.ref, e.parameter.decision);
    if (action === 'trash') return trash_(e.parameter.id);
    return json_({ ok: false, error: 'ไม่รู้จัก action: ' + action });
  } catch (err) {
    return json_({ ok: false, error: String(err.message || err) });
  }
}

// รายการ photo_id ที่ต้องซ่อน (ให้เซิร์ฟเวอร์ดึงทุก ~5 นาทีผ่าน REMOVED_URL)
function removed_() {
  const v = sheet_(REQUEST_SHEET).getDataRange().getValues(), h = v[0].map(String);
  const id = colIndex_(h, 'photo_id'), st = colIndex_(h, 'status'), out = [];
  for (var r = 1; r < v.length; r++) {
    var s = String(v[r][st]).trim();
    if (HIDE_STATUSES.some(function (a) { return s.indexOf(a) === 0; }) && idOf_(v[r][id])) out.push(idOf_(v[r][id]));
  }
  return json_({ removed: out });
}

function requests_() {
  const v = sheet_(REQUEST_SHEET).getDataRange().getValues(), h = v[0].map(String), idx = {}, rows = [];
  Object.keys(COLS).forEach(function (k) { idx[k] = colIndex_(h, k); });
  for (var r = v.length - 1; r >= 1 && rows.length < 300; r--) {       // ใหม่สุดก่อน
    var o = {};
    Object.keys(idx).forEach(function (k) { o[k] = cell_(v[r][idx[k]]); });
    if (/^DEL-/.test(o.ref)) rows.push(o);
  }
  return json_({ ok: true, rows: rows });
}

function history_(limit) {
  const v = sheet_(SEARCH_SHEET).getDataRange().getValues();
  if (!v.length) return json_({ ok: true, headers: [], rows: [] });
  const rows = [];
  for (var r = v.length - 1; r >= 1 && rows.length < Math.min(limit, 1000); r--) {
    var row = v[r].map(cell_);
    if (row.join('').trim()) rows.push(row);
  }
  return json_({ ok: true, headers: v[0].map(String), rows: rows });
}

function trashFile_(id) {
  DriveApp.getFileById(id).setTrashed(true);
}

function trash_(id) {
  id = idOf_(id);
  if (!id) return json_({ ok: false, error: 'id ไม่ถูกต้อง' });
  trashFile_(id);
  return json_({ ok: true, photo_id: id });
}

function setStatus_(ref, decision) {
  if (!/^DEL-\d{6}-[0-9A-Fa-f]{4}$/.test(String(ref))) return json_({ ok: false, error: 'ref ไม่ถูกต้อง' });
  const lock = LockService.getScriptLock(); lock.waitLock(20000);
  try {
    const sh = sheet_(REQUEST_SHEET), v = sh.getDataRange().getValues(), h = v[0].map(String);
    const cRef = colIndex_(h, 'ref'), cId = colIndex_(h, 'photo_id'), cSt = colIndex_(h, 'status');
    for (var r = 1; r < v.length; r++) {
      if (String(v[r][cRef]).trim() !== ref) continue;
      const pid = idOf_(v[r][cId]); var status, trashed = false, msg = '';
      if (decision === 'reject') {
        status = ST_REJECTED;
      } else if (decision === 'approve') {
        status = ST_APPROVED;
        try { trashFile_(pid); trashed = true; status = ST_DELETED; msg = 'ย้ายไฟล์ใน Drive เข้าถังขยะแล้ว'; }
        catch (err) { msg = 'อนุมัติแล้ว แต่ย้ายไฟล์เข้าถังขยะไม่สำเร็จ: ' + err.message + ' (รูปยังถูกซ่อนจากการค้นหา)'; }
      } else return json_({ ok: false, error: 'decision ต้องเป็น approve หรือ reject' });
      sh.getRange(r + 1, cSt + 1).setValue(status);
      return json_({ ok: true, status: status, trashed: trashed, photo_id: pid, message: msg });
    }
    return json_({ ok: false, error: 'ไม่พบคำขอ ' + ref + ' ในชีต' });
  } finally { lock.releaseLock(); }
}

// ── บันทึกการใช้งานแอดมิน ──
const ADMIN_LOG_HEADERS = ['เวลา', 'ผู้ดูแล', 'IP', 'การกระทำ', 'รายละเอียด', 'ผล'];
function safe_(v) { v = String(v === null || v === undefined ? '' : v); return /^[=+\-@]/.test(v) ? "'" + v : v; }   // กันสูตรแทรกในชีต

function adminLog_(d) {                       // เรียกจาก doPost (type = admin_log)
  var k = PropertiesService.getScriptProperties().getProperty('ADMIN_KEY');
  if (!k || d.key !== k) return json_({ ok: false, error: 'key ไม่ถูกต้อง' });
  var ss = SpreadsheetApp.getActive(), sh = ss.getSheetByName(ADMIN_LOG_SHEET);
  if (!sh) { sh = ss.insertSheet(ADMIN_LOG_SHEET); sh.appendRow(ADMIN_LOG_HEADERS); sh.setFrozenRows(1); }
  sh.appendRow([safe_(d.time), safe_(d.admin), safe_(d.ip), safe_(d.action), safe_(d.detail), safe_(d.result)]);
  return json_({ ok: true });
}

function adminLogRead_() {
  var sh = SpreadsheetApp.getActive().getSheetByName(ADMIN_LOG_SHEET);
  if (!sh) return json_({ ok: true, headers: ADMIN_LOG_HEADERS, rows: [] });
  var v = sh.getDataRange().getValues(), rows = [];
  for (var r = v.length - 1; r >= 1 && rows.length < 500; r--) rows.push(v[r].map(cell_));
  return json_({ ok: true, headers: ADMIN_LOG_HEADERS, rows: rows });
}

// รันฟังก์ชันนี้ 1 ครั้งจากตัวแก้ไข เพื่อให้ Google ขอสิทธิ์ Drive/Spreadsheet
function authorize_() { DriveApp.getRootFolder(); SpreadsheetApp.getActive(); }
