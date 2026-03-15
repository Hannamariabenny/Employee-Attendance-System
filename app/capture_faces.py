import cv2
import numpy as np
import face_recognition
from .database import supabase

async def register_employee_from_upload(name, photos):
    all_encodings = []

    for photo in photos:
        image_bytes = await photo.read()
        np_img = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if bgr is None:
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)

        locations = face_recognition.face_locations(rgb, model="hog")
        if not locations:
            continue

        encodings = face_recognition.face_encodings(rgb, known_face_locations=locations)
        if encodings:
            all_encodings.append(encodings[0])

    if not all_encodings:
        return "No faces detected in any of the uploaded photos — please try clearer images"

    if len(all_encodings) < 1:
        return "No valid faces detected — please try again with clearer photos"

    # ✅ Average all encodings into one robust encoding
    avg_encoding = np.mean(all_encodings, axis=0).tolist()

    supabase.table("employees").insert({
        "name": name,
        "embedding": avg_encoding,
        "photo_count": len(all_encodings)
    }).execute()

    return f"✅ Employee '{name}' registered using {len(all_encodings)} photos"