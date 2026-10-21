import json
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from jose import JWTError, jwt
import bcrypt

from app.ai_engine import build_engines

BASE = Path(__file__).parent.parent
DATA = BASE / "data"
DB = BASE / "career.db"
STATIC = BASE / "static"

SECRET_KEY = "faris_super_secret_key_change_in_production_2026"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24


# ---------- bcrypt direct (بدون passlib) ----------
def hash_password(password: str) -> str:
    """تشفير كلمة المرور عبر bcrypt مباشرة."""
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    """التحقق من كلمة المرور مقابل hash bcrypt."""
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


# ---------- نماذج ----------
class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class Profile(BaseModel):
    family: str
    skills: List[str] = []
    interest: str = ""
    experience: str = "none"
    sector: str = "any"

class ChatRequest(BaseModel):
    message: str

class JobPostCreate(BaseModel):
    title: str = Field(..., max_length=200)
    type: str = Field(..., max_length=100)
    details: str = Field(..., max_length=2000)
    contact: Optional[str] = Field("", max_length=200)
    specialty: Optional[str] = Field("", max_length=200)


# ---------- تهيئة قاعدة البيانات ----------
def init_db():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.executescript("""
        DROP TABLE IF EXISTS universities;
        DROP TABLE IF EXISTS colleges;
        DROP TABLE IF EXISTS departments;
        CREATE TABLE universities (id TEXT PRIMARY KEY, name TEXT);
        CREATE TABLE colleges (id TEXT PRIMARY KEY, name TEXT, university_id TEXT);
        CREATE TABLE departments (id TEXT PRIMARY KEY, name TEXT, college_id TEXT, family TEXT, branches TEXT);

        DROP TABLE IF EXISTS admin_users;
        CREATE TABLE admin_users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, hashed_password TEXT);

        DROP TABLE IF EXISTS job_posts;
        CREATE TABLE job_posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            type TEXT NOT NULL,
            details TEXT NOT NULL,
            contact TEXT,
            specialty TEXT,
            status TEXT DEFAULT 'pending',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
    """)

    with open(DATA / "universities.json", "r", encoding="utf-8") as f:
        unis = json.load(f)
    for i, item in enumerate(unis):
        u_name, colleges = item[0], item[1]
        uid = f"u{i}"
        cur.execute("INSERT INTO universities VALUES (?, ?)", (uid, u_name))
        for j, c in enumerate(colleges):
            c_name, depts = c[0], c[1]
            cid = f"{uid}_c{j}"
            cur.execute("INSERT INTO colleges VALUES (?, ?, ?)", (cid, c_name, uid))
            for k, d in enumerate(depts):
                d_name, family, branches = d[0], d[1], d[2]
                cur.execute("INSERT INTO departments VALUES (?, ?, ?, ?, ?)",
                            (f"{cid}_d{k}", d_name or c_name, cid, family, ",".join(branches)))

    # كلمة مرور المطور مشفّرة
    hashed_pw = hash_password("12300123")
    cur.execute("INSERT OR IGNORE INTO admin_users (username, hashed_password) VALUES (?, ?)",
                ("faris", hashed_pw))

    conn.commit()
    conn.close()


init_db()
RECOMMENDER, CHATBOT = build_engines()


# ---------- دوال JWT ----------
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verify_admin(authorization: Optional[str] = Header(None)) -> str:
    """التحقق من توكن JWT في الهيدر Authorization: Bearer <token>"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    token = authorization[7:]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token")
        conn = sqlite3.connect(DB)
        row = conn.execute("SELECT username FROM admin_users WHERE username=?", (username,)).fetchone()
        conn.close()
        if not row:
            raise HTTPException(status_code=401, detail="Admin not found")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


# ---------- التطبيق ----------
app = FastAPI(
    title="بوصلة الخريج - Smart Career Compass",
    description="نظام ذكي للتوصية بالوظائف لخريجي الجامعات العراقية",
    version="2.0.0",
)

app.mount("/static", StaticFiles(directory=STATIC), name="static")


# ---------- نقاط النهاية العامة ----------
@app.get("/", include_in_schema=False)
def root():
    return FileResponse(STATIC / "index.html")


@app.get("/api/universities")
def universities():
    conn = sqlite3.connect(DB)
    rows = conn.execute("SELECT id, name FROM universities ORDER BY name").fetchall()
    conn.close()
    return [{"id": r[0], "name": r[1]} for r in rows]


@app.get("/api/colleges/{uni_id}")
def colleges(uni_id: str):
    conn = sqlite3.connect(DB)
    rows = conn.execute("SELECT id, name FROM colleges WHERE university_id=? ORDER BY name", (uni_id,)).fetchall()
    conn.close()
    if not rows:
        raise HTTPException(404, "University not found")
    return [{"id": r[0], "name": r[1]} for r in rows]


@app.get("/api/departments/{col_id}")
def departments(col_id: str):
    conn = sqlite3.connect(DB)
    rows = conn.execute("SELECT id, name, family FROM departments WHERE college_id=? ORDER BY name", (col_id,)).fetchall()
    conn.close()
    return [{"id": r[0], "name": r[1], "family": r[2]} for r in rows]


@app.get("/api/search")
def search(q: str):
    conn = sqlite3.connect(DB)
    like = f"%{q}%"
    results = []
    for r in conn.execute("SELECT name FROM universities WHERE name LIKE ?", (like,)):
        results.append({"type": "جامعة", "name": r[0]})
    for r in conn.execute("SELECT c.name, u.name FROM colleges c JOIN universities u ON u.id=c.university_id WHERE c.name LIKE ?", (like,)):
        results.append({"type": "كلية", "name": r[0], "parent": r[1]})
    for r in conn.execute("SELECT d.name, c.name, u.name FROM departments d JOIN colleges c ON c.id=d.college_id JOIN universities u ON u.id=c.university_id WHERE d.name LIKE ?", (like,)):
        results.append({"type": "قسم", "name": r[0], "parent": f"{r[1]} - {r[2]}"})
    conn.close()
    return results[:30]


@app.get("/api/skills")
def skills():
    with open(DATA / "skills.json", "r", encoding="utf-8") as f:
        return json.load(f)


@app.post("/api/recommend")
def recommend(profile: Profile):
    results = RECOMMENDER.recommend(profile.model_dump(), top_k=10)
    return {"count": len(results), "recommendations": results}


@app.post("/api/chat")
def chat(req: ChatRequest):
    return CHATBOT.respond(req.message)


@app.get("/api/jobs")
def all_jobs():
    return RECOMMENDER.jobs


# ---------- إعلانات الوظائف العامة ----------
@app.get("/api/job-posts")
def get_approved_job_posts():
    """الإعلانات المعتمدة فقط (للزوار)"""
    conn = sqlite3.connect(DB)
    rows = conn.execute("""
        SELECT id, title, type, details, contact, specialty, created_at
        FROM job_posts WHERE status='approved'
        ORDER BY created_at DESC
    """).fetchall()
    conn.close()
    return [{
        "id": r[0], "title": r[1], "type": r[2], "details": r[3],
        "contact": r[4], "specialty": r[5], "created_at": r[6]
    } for r in rows]


@app.post("/api/job-posts")
def create_job_post(post: JobPostCreate):
    """إنشاء إعلان جديد (يبدأ بحالة pending)"""
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO job_posts (title, type, details, contact, specialty)
        VALUES (?, ?, ?, ?, ?)
    """, (post.title, post.type, post.details, post.contact, post.specialty))
    conn.commit()
    post_id = cur.lastrowid
    conn.close()
    return {
        "id": post_id,
        "status": "pending",
        "message": "تم استلام إعلانك، وسيتم مراجعته من قبل الإدارة قبل نشره."
    }


# ---------- مصادقة المطور ----------
@app.post("/api/admin/login", response_model=TokenResponse)
def admin_login(login: LoginRequest):
    conn = sqlite3.connect(DB)
    row = conn.execute("SELECT username, hashed_password FROM admin_users WHERE username=?",
                       (login.username,)).fetchone()
    conn.close()
    if not row or not verify_password(login.password, row[1]):
        raise HTTPException(status_code=401, detail="اسم المستخدم أو كلمة المرور غير صحيحة")
    token = create_access_token({"sub": login.username})
    return {"access_token": token}


# ---------- لوحة التحكم (للمطور) ----------
@app.get("/api/admin/job-posts")
def admin_list_job_posts(admin: str = Depends(verify_admin)):
    conn = sqlite3.connect(DB)
    pending = conn.execute("""
        SELECT id, title, type, details, contact, specialty, created_at
        FROM job_posts WHERE status='pending' ORDER BY created_at DESC
    """).fetchall()
    approved = conn.execute("""
        SELECT id, title, type, details, contact, specialty, created_at
        FROM job_posts WHERE status='approved' ORDER BY created_at DESC
    """).fetchall()
    conn.close()

    def to_dict(r):
        return {
            "id": r[0], "title": r[1], "type": r[2], "details": r[3],
            "contact": r[4], "specialty": r[5], "created_at": r[6]
        }

    return {
        "pending": [to_dict(r) for r in pending],
        "approved": [to_dict(r) for r in approved]
    }


@app.post("/api/admin/approve/{post_id}")
def approve_job_post(post_id: int, admin: str = Depends(verify_admin)):
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("UPDATE job_posts SET status='approved' WHERE id=?", (post_id,))
    if cur.rowcount == 0:
        conn.close()
        raise HTTPException(404, "Post not found")
    conn.commit()
    conn.close()
    return {"status": "approved", "id": post_id}


@app.delete("/api/admin/delete/{post_id}")
def delete_job_post(post_id: int, admin: str = Depends(verify_admin)):
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("DELETE FROM job_posts WHERE id=?", (post_id,))
    if cur.rowcount == 0:
        conn.close()
        raise HTTPException(404, "Post not found")
    conn.commit()
    conn.close()
    return {"status": "deleted", "id": post_id}