import cv2
import numpy as np
import face_recognition
from .database import supabase

async def register_employee_from_upload(name, photo):
    image_bytes = await photo.read()
    np_img = np.frombuffer(image_bytes, dtype=np.uint8)

    bgr = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    if bgr is None:
        return "Invalid image file"

    # ✅ Safe BGR -> RGB conversion
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # ✅ Make sure array is contiguous + correct dtype for dlib
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)

    # ✅ Detect face locations first (more stable)
    locations = face_recognition.face_locations(rgb, model="hog")

    if not locations:
        return "No face detected in photo"

    encodings = face_recognition.face_encodings(rgb, known_face_locations=locations)

    if not encodings:
        return "Face found, but encoding failed (try a clearer photo)"

    encoding = np.array(encodings[0], dtype=np.float64).tolist()

    supabase.table("employees").insert({
        "name": name,
        "embedding": encoding
    }).execute()

    return f"Employee {name} registered successfully"
