"""Offline Math Tutor backend.  Run:  python -m uvicorn main:app --port 8000
Then open http://localhost:8000"""
import hashlib
import json
import re
import secrets
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ollama
import sympy as sp
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sympy.parsing.sympy_parser import (
    convert_xor, implicit_multiplication_application, parse_expr, standard_transformations,
)

BASE = Path(__file__).parent
DB = BASE / "app.db"
DEFAULT_MODEL = "qwen2.5:3b"
COURSE_FILES = {
    "Inter 1st year": "inter_1.json", "Inter 2nd year": "inter_2.json",
    "Diploma": "diploma.json", "B.Tech": "btech.json",
}

app = FastAPI()
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")


# ---------- database ----------
def db():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c


with db() as _c:
    _c.executescript("""
    CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, username TEXT UNIQUE, salt TEXT, pw TEXT, settings TEXT DEFAULT '{}');
    CREATE TABLE IF NOT EXISTS sessions(token TEXT PRIMARY KEY, user_id INTEGER);
    CREATE TABLE IF NOT EXISTS chats(id INTEGER PRIMARY KEY, user_id INTEGER, title TEXT, context TEXT);
    CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY, chat_id INTEGER, role TEXT, content TEXT);
    """)


def hash_pw(pw, salt):
    return hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), 200_000).hex()


def user(request: Request):
    tok = request.cookies.get("sid")
    with db() as c:
        r = c.execute("SELECT u.* FROM sessions s JOIN users u ON u.id=s.user_id WHERE s.token=?", (tok,)).fetchone()
    if not r:
        raise HTTPException(401, "Not logged in")
    return r


def start_session(response: Response, uid: int):
    tok = secrets.token_hex(24)
    with db() as c:
        c.execute("INSERT INTO sessions VALUES(?,?)", (tok, uid))
    response.set_cookie("sid", tok, httponly=True, samesite="lax", max_age=60 * 60 * 24 * 30)


# ---------- auth ----------
class Cred(BaseModel):
    username: str
    password: str


@app.post("/api/register")
def register(b: Cred, response: Response):
    name = b.username.strip()
    if not (3 <= len(name) <= 30) or len(b.password) < 6:
        raise HTTPException(400, "Username 3-30 characters, password at least 6 characters")
    salt = secrets.token_hex(8)
    try:
        with db() as c:
            cur = c.execute("INSERT INTO users(username,salt,pw) VALUES(?,?,?)", (name, salt, hash_pw(b.password, salt)))
            uid = cur.lastrowid
    except sqlite3.IntegrityError:
        raise HTTPException(409, "That username is taken")
    start_session(response, uid)
    return {"username": name}


@app.post("/api/login")
def login(b: Cred, response: Response):
    with db() as c:
        r = c.execute("SELECT * FROM users WHERE username=?", (b.username.strip(),)).fetchone()
    if not r or not secrets.compare_digest(r["pw"], hash_pw(b.password, r["salt"])):
        raise HTTPException(401, "Wrong username or password")
    start_session(response, r["id"])
    return {"username": r["username"]}


@app.post("/api/logout")
def logout(request: Request, response: Response):
    with db() as c:
        c.execute("DELETE FROM sessions WHERE token=?", (request.cookies.get("sid"),))
    response.delete_cookie("sid")
    return {"ok": True}


@app.get("/api/me")
def me(u=Depends(user)):
    return {"username": u["username"]}


# ---------- settings & syllabus ----------
@app.get("/api/settings")
def get_settings(u=Depends(user)):
    return json.loads(u["settings"] or "{}")


@app.post("/api/settings")
def set_settings(body: dict, u=Depends(user)):
    clean = {k: str(body.get(k, ""))[:60] for k in ("model", "lang", "theme")}
    with db() as c:
        c.execute("UPDATE users SET settings=? WHERE id=?", (json.dumps(clean), u["id"]))
    return clean


@app.get("/api/syllabus")
def syllabus(u=Depends(user)):
    out = {}
    for name, f in COURSE_FILES.items():
        try:
            out[name] = json.loads((BASE / "data" / f).read_text(encoding="utf-8"))
        except Exception:
            out[name] = []
    return out


# ---------- chats ----------
class NewChat(BaseModel):
    context: dict = {}


class Ask(BaseModel):
    text: str
    context: dict = {}


def own_chat(cid, u):
    with db() as c:
        r = c.execute("SELECT * FROM chats WHERE id=? AND user_id=?", (cid, u["id"])).fetchone()
    if not r:
        raise HTTPException(404, "Chat not found")
    return r


@app.get("/api/chats")
def list_chats(u=Depends(user)):
    with db() as c:
        rows = c.execute("SELECT id,title FROM chats WHERE user_id=? ORDER BY id DESC", (u["id"],)).fetchall()
    return [dict(r) for r in rows]


@app.post("/api/chats")
def new_chat(b: NewChat, u=Depends(user)):
    with db() as c:
        cur = c.execute("INSERT INTO chats(user_id,title,context) VALUES(?,?,?)", (u["id"], "New chat", json.dumps(b.context)))
    return {"id": cur.lastrowid}


@app.get("/api/chats/{cid}")
def get_chat(cid: int, u=Depends(user)):
    chat = own_chat(cid, u)
    with db() as c:
        rows = c.execute("SELECT role,content FROM messages WHERE chat_id=? ORDER BY id", (cid,)).fetchall()
    msgs = [{"role": r["role"], "content": json.loads(r["content"]) if r["role"] == "assistant" else r["content"]} for r in rows]
    return {"context": json.loads(chat["context"] or "{}"), "messages": msgs}


@app.delete("/api/chats/{cid}")
def del_chat(cid: int, u=Depends(user)):
    own_chat(cid, u)
    with db() as c:
        c.execute("DELETE FROM messages WHERE chat_id=?", (cid,))
        c.execute("DELETE FROM chats WHERE id=?", (cid,))
    return {"ok": True}


@app.delete("/api/chats")
def del_all(u=Depends(user)):
    with db() as c:
        c.execute("DELETE FROM messages WHERE chat_id IN (SELECT id FROM chats WHERE user_id=?)", (u["id"],))
        c.execute("DELETE FROM chats WHERE user_id=?", (u["id"],))
    return {"ok": True}


# ---------- maths check (SymPy) ----------
TF = standard_transformations + (implicit_multiplication_application, convert_xor)
X = sp.Symbol("x")
LOCALS = {"x": X, "e": sp.E, "ln": sp.log}
KEYS = [
    ("diff", r"differentiate|derivative of|derivative|d/dx"),
    ("int", r"integrate|integral of|integration of|integral"),
    ("solve", r"solve"),
    ("simp", r"simplify"),
    ("lim", r"limit of|limit"),
]
_pool = ThreadPoolExecutor(2)


def P(s):
    return parse_expr(s, local_dict=LOCALS, transformations=TF)


def clean(s):
    s = re.sub(r"(with respect to x|w\.r\.t\.? x|wrt x)\s*$", "", s.strip(), flags=re.I)
    return re.sub(r"^(?:\s|of|the|function|:)+", "", s, flags=re.I).strip().rstrip("?.")


def compute(text):
    best = None
    for kind, pat in KEYS:
        m = re.search(pat, text, re.I)
        if m and (best is None or m.start() < best[1].start()):
            best = (kind, m)
    if not best:
        return None
    kind, m = best
    rest = text[m.end():]
    if kind == "lim":
        pt = 0
        mm = re.search(r"x\s*(?:->|→|tends to|approaches)\s*([-\w.]+)", rest, re.I)
        if mm:
            pt = P(mm.group(1))
            rest = re.sub(r"\b(as|when)\b", "", rest.replace(mm.group(0), ""), flags=re.I)
        return sp.limit(P(clean(rest)), X, pt)
    rest = clean(rest)
    if kind == "solve":
        if "=" in rest:
            left, right = rest.split("=", 1)
            return sp.solve(P(left) - P(right), X)
        return sp.solve(P(rest), X)
    e = P(rest)
    if kind == "diff":
        return sp.diff(e, X)
    if kind == "int":
        return sp.integrate(e, X)
    return sp.simplify(e)


def try_sympy(text):
    """Returns (text, latex) of the verified result, or None if the question is not a simple recognised type."""
    try:
        r = _pool.submit(compute, text).result(timeout=10)
        return None if r is None else (str(r), sp.latex(r))
    except Exception:
        return None


# ---------- explanation (Ollama) ----------
PROMPT = r"""You are a mathematics tutor for <<course>> students in Telangana, India.
Subject: <<subject>>. Unit: <<unit>>. Year/Semester: <<year>> / <<sem>>.
Answer ONLY mathematics questions. For anything else reply with just:
===SOLUTION===
I only answer mathematics questions for the selected course.

STRICT RULES:
- Write ALL maths in LaTeX inside \( ... \) (for example \(\dfrac{x^2}{2}\), \(\int \cos^2 x \, dx\), \(\sqrt{x}\)). Never use Markdown, bold, or $ signs.
- Labels (Given, To find, Formula, Step 1, Final Answer) are plain words outside the \( \) brackets.
- One equation per line. Start continuing lines with = inside the brackets.
- Reply with exactly three sections, each starting with its marker line. The SOLUTION section must never be empty.
- If a VERIFIED RESULT is given, your Final Answer must equal it exactly. If it is a list for an equation (like [2, 3]), those are the roots: write \(x = 2\) or \(x = 3\), never the raw list.

EXAMPLE (follow this layout exactly)
Question: Integrate cos^2 x
===SOLUTION===
Given: \(I = \int \cos^2 x \, dx\)
Formula: \(\cos 2x = 2\cos^2 x - 1 \;\Rightarrow\; \cos^2 x = \dfrac{1+\cos 2x}{2}\)
Step 1: \(I = \int \dfrac{1+\cos 2x}{2} \, dx\)
Step 2: \(= \dfrac{1}{2}\int (1+\cos 2x)\, dx\)
Step 3: \(= \dfrac{1}{2}\int dx + \dfrac{1}{2}\int \cos 2x \, dx\)
Step 4: \(= \dfrac{x}{2} + \dfrac{1}{4}\sin 2x + C\)
Final Answer: \(I = \dfrac{x}{2} + \dfrac{1}{4}\sin 2x + C\)
===EXPLANATION===
Line 1: We want the integral of \(\cos^2 x\), which cannot be integrated directly.
Line 2: We use the identity \(\cos 2x = 2\cos^2 x - 1\) to rewrite \(\cos^2 x\) without a square.
Line 3: Substituting it gives \(\dfrac{1+\cos 2x}{2}\) inside the integral.
Line 4: We take the constant \(\dfrac{1}{2}\) outside and split the integral into two simple parts.
Line 5: \(\int dx = x\) and \(\int \cos 2x \, dx = \dfrac{\sin 2x}{2}\), then we add the constant \(C\).
===FORMULAS===
\(\cos 2x = 2\cos^2 x - 1\)
\(\int \cos ax \, dx = \dfrac{\sin ax}{a} + C\)

<<lang>>"""

SOLUTION_ONLY = r"""Write ONLY the written exam-notebook solution for the problem: Given, To find, Formula, numbered Steps (one equation per line), Final Answer.
Write all maths in LaTeX inside \( ... \). Labels are plain words outside the brackets. If a verified result is given, the Final Answer must equal it."""


EXPLAIN_ONLY = r"""You are given a solved maths problem. Write exactly two sections.
===EXPLANATION===
Plain-text paragraph lines, one per line of the solution, written as "Line 1: ...", "Line 2: ..." and explaining WHY each step is done in simple words. Maths inside \( ... \) in LaTeX.
===FORMULAS===
Only the formulas or identities the solution used, one per line, each in LaTeX inside \( ... \).
<<lang>>"""


def tidy(t):
    """Remove Markdown noise the model sometimes adds; the maths itself stays as LaTeX."""
    return re.sub(r"```\w*", "", t).replace("**", "").strip()


def parse_sections(t):
    """Split the model reply into sections. Tolerates marker variants such as '=== FORMULAS', '### EXPLANATION:' or a bare 'FORMULAS' line."""
    keys = {"SOLUTION": "solution", "EXPLANATION": "explanation", "FORMULAS": "formulas"}
    out, cur = {v: "" for v in keys.values()}, None
    for line in t.splitlines():
        s = line.strip()
        word = s.strip("=#*: ").upper()
        if word in keys and len(s) <= 40:
            cur = keys[word]
        elif re.fullmatch(r"=+", s):
            continue
        elif cur:
            out[cur] += line + "\n"
    if not any(v.strip() for v in out.values()):
        out["solution"] = t
    return {k: v.strip() for k, v in out.items()}


@app.post("/api/chats/{cid}/ask")
def ask(cid: int, b: Ask, u=Depends(user)):
    chat = own_chat(cid, u)
    text = b.text.strip()[:2000]
    st = json.loads(u["settings"] or "{}")
    ctx = b.context
    v = try_sympy(text)
    verified, vtex = (v if v else (None, None))
    lang = "Write the EXPLANATION in Telugu (keep maths symbols as they are)." if st.get("lang") == "Telugu" else ""
    system = PROMPT
    for k, val in {"course": ctx.get("course", "not selected"), "subject": ctx.get("subject", "not selected"),
                   "unit": ctx.get("unit", "not selected"), "year": ctx.get("year", "-"), "sem": ctx.get("sem", "-"), "lang": lang}.items():
        system = system.replace(f"<<{k}>>", str(val))
    user_msg = text + (f"\n\nVERIFIED RESULT (from math engine): {verified}  (in LaTeX: {vtex})" if verified else "")
    model = st.get("model") or DEFAULT_MODEL

    def call_llm(sys_prompt, usr):
        r = ollama.chat(model=model, messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": usr}])
        return r["message"]["content"]

    try:
        ans = {k: tidy(v) for k, v in parse_sections(call_llm(system, user_msg)).items()}
        if not ans["solution"]:  # model skipped the solution: ask once more for just that part
            hint = f"Problem: {text}\n" + (f"Verified result: {verified} (LaTeX: {vtex})\n" if verified else "") + f"Explanation: {ans['explanation']}"
            ans["solution"] = tidy(parse_sections(call_llm(SOLUTION_ONLY, hint))["solution"])
    except Exception as e:
        raise HTTPException(503, f"Could not get an answer from Ollama. Is it running, and is the model name in Settings correct? ({str(e)[:150]})")
    if ans["solution"] and (not ans["explanation"] or not ans["formulas"]):
        try:  # model stopped after the solution: ask once for the missing parts
            extra = parse_sections(call_llm(EXPLAIN_ONLY.replace("<<lang>>", lang), f"Problem: {text}\n\nSolution:\n{ans['solution']}"))
            for k in ("explanation", "formulas"):
                if not ans[k]:
                    ans[k] = tidy(extra[k]) if k in extra and extra[k] != ans["solution"] else ""
        except Exception:
            pass
    if not ans["solution"]:
        ans["solution"] = f"Final Answer: \\({vtex}\\)" if verified else "No solution was generated. Please ask again."
    ans["verified"] = verified
    ans["verified_tex"] = vtex
    with db() as c:
        c.execute("INSERT INTO messages(chat_id,role,content) VALUES(?,?,?)", (cid, "user", text))
        c.execute("INSERT INTO messages(chat_id,role,content) VALUES(?,?,?)", (cid, "assistant", json.dumps(ans)))
        c.execute("UPDATE chats SET context=? WHERE id=?", (json.dumps(ctx), cid))
        if chat["title"] == "New chat":
            c.execute("UPDATE chats SET title=? WHERE id=?", (text[:40], cid))
    return ans


@app.get("/")
def index():
    return FileResponse(BASE / "index.html")
