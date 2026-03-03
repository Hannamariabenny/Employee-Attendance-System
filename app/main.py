from fastapi import FastAPI, UploadFile, File, Form
from .capture_faces import register_employee_from_upload
from .recognize_face import recognize_and_mark
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from .database import supabase
from pydantic import BaseModel

app = FastAPI()

templates = Jinja2Templates(directory="templates")

class AttendanceRequest(BaseModel):
    image: str

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.post("/register/")
async def register_employee(name: str = Form(...), photo: UploadFile = File(...)):
    result = await register_employee_from_upload(name, photo)
    return {"result": result}


@app.post("/attendance/")
async def attendance(data: AttendanceRequest):
    image_base64 = data.image

    image_data = base64.b64decode(image_base64.split(",")[1])
    image = Image.open(BytesIO(image_data))
    frame = np.array(image)

    result = recognize_and_mark(frame)

    return {"result": result}

@app.get("/employees/")
def get_employees():
    response = (
        supabase
        .table("employees")
        .select("id,name")
        .order("name", desc=False)   # ✅ Alphabetical order
        .execute()
    )
    return response.data


@app.get("/records/")
def get_records():
    data = (
        supabase
        .table("attendance")
        .select("id, timestamp, employee_id, employees(name)")
        .order("timestamp", desc=True)
        .execute()
        .data
    )

    # flatten employees(name) into "name"
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
