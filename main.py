# main.py  –  MCP-style interactive PDF form filler (file upload version)
import os, io, uuid, json, base64
from typing import Dict, List
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pdf2image import convert_from_path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from pypdf import PdfReader, PdfWriter
from openai import OpenAI

'''
HIGH LEVEL FLOW:
POST /contexts            (PDF -> new session)
POST /contexts/{id}/steps (model analyses PDF -> returns missing fields -> asks user)
POST /contexts/{id}/messages (user supplies answers)
… loop …
POST /contexts/{id}/steps (all answers present -> builds overlay -> returns filled PDF)
GET  /contexts/{id}       (inspect state anytime)

USE LOCAL CLI:
curl -X POST http://localhost:8000/contexts \
     -F "pdf=@form.pdf"

Response: {"id":"3f7c5635-23a1-48ac-a848-ce95918ea5eb","missing":["Name","Email","Birthday"]}%
VERY GOOD!

curl -X POST http://localhost:8000/contexts/3f7c5635-23a1-48ac-a848-ce95918ea5eb/messages \
     -H "Content-Type: application/json" \
     -d '{
           "answers": {
             "Name":     "Aidan Burrowes",
             "Email":    "aidan@example.com",
             "Birthday": "1990-01-01"
           }
         }'

 Response: {"accepted":["Name","Email","Birthday"],"missing":[]}% 
 YES!

 # request the next step (server will (should...) stream the completed PDF)
curl -X POST \
     http://localhost:8000/contexts/3f7c5635-23a1-48ac-a848-ce95918ea5eb/steps \
     --output filled_form.pdf
'''

# ─── Config ────────────────────────────────────────────────
client       = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
VISION_MODEL = "gpt-4o-2024-05-13"
# ───────────────────────────────────────────────────────────

app = FastAPI()
SESSIONS: Dict[str, dict] = {}              # in-memory session storage

# ─── Helper functions ─────────────────────────────────────
def pdf_page_png(pdf_bytes: bytes, dpi=150) -> bytes:
    tmp = NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(pdf_bytes); tmp.close()
    img = convert_from_path(tmp.name, dpi=dpi, first_page=1, last_page=1)[0]
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return buf.getvalue()

def vision_extract_fields(pdf_bytes: bytes) -> List[dict]:
    img_b64 = base64.b64encode(pdf_page_png(pdf_bytes)).decode()
    prompt  = (
        "Look at this PDF form page (8.5×11 inches, origin bottom-left). "
        "List every field a user should fill. "
        "Return ONLY JSON array; each item must have label,x,y keys. "
    )
    resp = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role":"user","content":[
                {"type":"text","text":prompt},
                {"type":"image_url","image_url":{"url":f"data:image/png;base64,{img_b64}"}}
            ]}
        ],
        max_tokens=400
    )
    txt = resp.choices[0].message.content.strip()
    if txt.startswith("```"):                       # strip markdown fence
        txt = txt.split("```")[1].strip()
    return json.loads(txt)

def overlay_values(plan: List[dict], answers: Dict[str,str], base_pdf: bytes) -> bytes:
    overlay_path = NamedTemporaryFile(delete=False, suffix=".pdf").name
    c = canvas.Canvas(overlay_path, pagesize=letter)
    for item in plan:
        lbl = item["label"]
        if lbl in answers:
            c.drawString(float(item["x"]), float(item["y"]), answers[lbl])
    c.save()

    base_tmp = NamedTemporaryFile(delete=False, suffix=".pdf"); base_tmp.write(base_pdf); base_tmp.close()
    out_path = NamedTemporaryFile(delete=False, suffix=".pdf").name

    base = PdfReader(base_tmp.name); over = PdfReader(overlay_path)
    writer = PdfWriter()
    page = base.pages[0]; page.merge_page(over.pages[0]); writer.add_page(page)
    with open(out_path, "wb") as f: writer.write(f)
    with open(out_path, "rb") as f: return f.read()

# ─── Endpoints ─────────────────────────────────────────────
@app.post("/contexts")                          # 1️⃣  upload PDF
async def create_context(pdf: UploadFile = File(...)):
    if not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Upload a PDF")
    pdf_bytes = await pdf.read()

    try:
        plan = vision_extract_fields(pdf_bytes)   # [{"label","x","y"}, ...]
    except Exception as e:
        raise HTTPException(502, f"Vision model failed: {e}")

    ctx_id = str(uuid.uuid4())
    SESSIONS[ctx_id] = {"pdf": pdf_bytes, "plan": plan, "answers": {}}

    missing = [p["label"] for p in plan]
    return {"id": ctx_id, "missing": missing}

@app.post("/contexts/{cid}/messages")           # 2️⃣  supply answers
async def add_answers(cid: str, payload: Dict[str, Dict[str,str]]):
    ctx = SESSIONS.get(cid);                      # payload {"answers": {...}}
    if not ctx: raise HTTPException(404, "context not found")
    if "answers" not in payload:
        raise HTTPException(400, "missing 'answers' key")

    ctx["answers"].update(payload["answers"])
    still_missing = [p["label"] for p in ctx["plan"] if p["label"] not in ctx["answers"]]
    return {"accepted": list(payload["answers"].keys()), "missing": still_missing}

@app.post("/contexts/{cid}/steps")              # 3️⃣  progress / finish
async def step(cid: str):
    ctx = SESSIONS.get(cid)
    if not ctx: raise HTTPException(404, "context not found")
    missing = [p["label"] for p in ctx["plan"] if p["label"] not in ctx["answers"]]

    if missing:
        return JSONResponse(
            status_code=206,          # Partial Content – need more
            content={
                "status": "need_info",
                "missing": missing,
                "message": f"Please supply values for: {', '.join(missing)}"
            }
        )

    filled_pdf = overlay_values(ctx["plan"], ctx["answers"], ctx["pdf"])

    tmp_out = NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp_out.write(filled_pdf)
    tmp_out.close()

    return FileResponse(
        path=tmp_out.name,
        media_type="application/pdf",
        filename="filled_form.pdf"
    )


@app.get("/contexts/{cid}")                    # inspector (optional)
def get_context(cid: str):
    ctx = SESSIONS.get(cid) or {}
    return {
        "plan": ctx.get("plan"),
        "answers": ctx.get("answers"),
        "missing":[p["label"] for p in ctx.get("plan",[]) if p["label"] not in ctx.get("answers",{})]
    }

# ─── Dev runner ────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn, sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
