import cv2
import face_recognition
import numpy as np
from datetime import datetime
from .database import supabase
from datetime import datetime, date, timedelta

def recognize_and_mark(frame):
    # ✅ Safe BGR -> RGB conversion (avoids dlib TypeError)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame, model="hog")

    encodings = face_recognition.face_encodings(
        rgb_frame,
        known_face_locations=face_locations
    )

    if not encodings:
        return "No face detected"

    employees = supabase.table("employees").select("id,name,embedding").execute().data

    for unknown_encoding in encodings:
        unknown_encoding = np.array(unknown_encoding, dtype=np.float64)

        for emp in employees:
            stored_encoding = np.array(emp["embedding"], dtype=np.float64)

            # Safety check: face_recognition encodings are length 128
            if stored_encoding.shape[0] != 128:
                continue

            distance = np.linalg.norm(unknown_encoding - stored_encoding)

            if distance < 0.6:

                # ✅ Check if already marked today
                from datetime import datetime, timedelta

                # ✅ Check if already marked in last 2 minutes
                now = datetime.now()
                two_min_ago = (now - timedelta(minutes=2)).isoformat()
                now_iso = now.isoformat()

                existing = (
                    supabase.table("attendance")
                    .select("id")
                    .eq("employee_id", emp["id"])
                    .gte("timestamp", two_min_ago)
                    .lte("timestamp", now_iso)
                    .execute()
                    .data
                )

                if existing:
                    return f"Already marked in last 2 minutes for {emp['name']}"


                # ✅ Mark attendance only if not already present
                supabase.table("attendance").insert({
                    "employee_id": emp["id"],
                    "timestamp": datetime.now().isoformat()
                }).execute()

                return f"Attendance marked for {emp['name']}"


    return "Unknown Person"
