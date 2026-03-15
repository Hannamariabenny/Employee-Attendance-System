from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from .capture_faces import register_employee_from_upload
from .recognize_face import recognize_and_mark
from .database import supabase
from pydantic import BaseModel
import base64
from io import BytesIO
from PIL import Image
import numpy as np

app = FastAPI()

# ✅ CORS — allows frontend to call API freely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

class AttendanceRequest(BaseModel):
    image: str

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.post("/register/")
async def register_employee(name: str = Form(...), photos: list[UploadFile] = File(...)):
    if len(photos) == 0:
        return {"result": "Please provide at least 1 photo"}
    if len(photos) > 20:
        return {"result": "Maximum 20 photos allowed"}
    result = await register_employee_from_upload(name, photos)
    return {"result": result}

@app.post("/attendance/")
async def attendance(data: AttendanceRequest):
    image_data = base64.b64decode(data.image.split(",")[1])
    image = Image.open(BytesIO(image_data)).convert("RGB")  # ✅ Force RGB
    frame = np.array(image)
    result = recognize_and_mark(frame)
    return result  # ✅ Returns {"results": [...], "message": "..."}

@app.get("/employees/")
def get_employees():
    response = (
        supabase.table("employees")
        .select("id, name, photo_count, created_at")
        .order("name")
        .execute()
    )
    return response.data

@app.delete("/employees/{employee_id}")
def delete_employee(employee_id: int):
    # ✅ Delete attendance records first (foreign key constraint)
    supabase.table("attendance").delete().eq("employee_id", employee_id).execute()
    # ✅ Then delete the employee
    supabase.table("employees").delete().eq("id", employee_id).execute()
    return {"result": f"Employee {employee_id} deleted successfully"}

@app.get("/records/")
def get_records():
    data = (
        supabase.table("attendance")
        .select("id, timestamp, employee_id, employees(name)")
        .order("timestamp", desc=True)
        .execute()
        .data
    )
    result = []
    for row in data:
        emp = row.get("employees") or {}
        result.append({
            "id": row["id"],
            "employee_id": row["employee_id"],
            "name": emp.get("name"),
            "timestamp": row["timestamp"]
        })
    return result