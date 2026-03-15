import face_recognition
import numpy as np
from datetime import datetime, timedelta
from .database import supabase

# ✅ Per-person cooldown tracker (in memory)
cooldown_tracker = {}
COOLDOWN_MINUTES = 2

def is_on_cooldown(employee_id):
    if employee_id not in cooldown_tracker:
        return False
    last_marked = cooldown_tracker[employee_id]
    return datetime.now() - last_marked < timedelta(minutes=COOLDOWN_MINUTES)

def set_cooldown(employee_id):
    cooldown_tracker[employee_id] = datetime.now()

def recognize_and_mark(frame):
    # ✅ Frame comes in as RGB from PIL — no conversion needed
    rgb_frame = np.ascontiguousarray(frame, dtype=np.uint8)

    # ✅ Detect ALL faces in the frame
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")

    if not face_locations:
        return {"results": [], "message": "No face detected"}

    # ✅ Get encodings for ALL detected faces at once
    all_encodings = face_recognition.face_encodings(rgb_frame, known_face_locations=face_locations)

    if not all_encodings:
        return {"results": [], "message": "Faces found but encoding failed"}

    # ✅ Load all employees once (efficient — not per face)
    employees = supabase.table("employees").select("id,name,embedding").execute().data

    if not employees:
        return {"results": [], "message": "No employees registered yet"}

    # Pre-convert all stored embeddings once
    stored = []
    for emp in employees:
        arr = np.array(emp["embedding"], dtype=np.float64)
        if arr.shape[0] == 128:
            stored.append({"id": emp["id"], "name": emp["name"], "encoding": arr})

    results = []

    # ✅ Process each detected face independently
    for i, (unknown_encoding, location) in enumerate(zip(all_encodings, face_locations)):
        unknown_enc = np.array(unknown_encoding, dtype=np.float64)

        best_match = None
        best_distance = 0.5  # threshold — lower = stricter

        for emp in stored:
            distance = np.linalg.norm(unknown_enc - emp["encoding"])
            print(f"[DEBUG] Face {i+1} vs {emp['name']}: {round(distance, 4)}")
            if distance < best_distance:
                best_distance = distance
                best_match = emp

        top, right, bottom, left = location

        if best_match:
            emp_id = best_match["id"]
            name = best_match["name"]

            # ✅ Per-person in-memory cooldown check (fast)
            if is_on_cooldown(emp_id):
                results.append({
                    "name": name,
                    "status": "cooldown",
                    "message": f"Already marked recently",
                    "location": {"top": top, "right": right, "bottom": bottom, "left": left}
                })
                continue

            # ✅ Supabase duplicate check
            now = datetime.now()
            two_min_ago = (now - timedelta(minutes=COOLDOWN_MINUTES)).isoformat()
            existing = (
                supabase.table("attendance")
                .select("id")
                .eq("employee_id", emp_id)
                .gte("timestamp", two_min_ago)
                .execute()
                .data
            )

            if existing:
                set_cooldown(emp_id)
                results.append({
                    "name": name,
                    "status": "cooldown",
                    "message": "Already marked recently",
                    "location": {"top": top, "right": right, "bottom": bottom, "left": left}
                })
                continue

            # ✅ Mark attendance
            supabase.table("attendance").insert({
                "employee_id": emp_id,
                "timestamp": now.isoformat()
            }).execute()

            set_cooldown(emp_id)

            results.append({
                "name": name,
                "status": "marked",
                "message": f"Attendance marked for {name}",
                "location": {"top": top, "right": right, "bottom": bottom, "left": left}
            })

        else:
            results.append({
                "name": "Unknown",
                "status": "unknown",
                "message": "Unknown person",
                "location": {"top": top, "right": right, "bottom": bottom, "left": left}
            })

    # Build summary message
    marked = [r for r in results if r["status"] == "marked"]
    cooldowns = [r for r in results if r["status"] == "cooldown"]
    unknown = [r for r in results if r["status"] == "unknown"]

    summary_parts = []
    if marked:    summary_parts.append(f"✅ Marked: {', '.join(r['name'] for r in marked)}")
    if cooldowns: summary_parts.append(f"⏸ Already marked: {', '.join(r['name'] for r in cooldowns)}")
    if unknown:   summary_parts.append(f"❓ Unknown: {len(unknown)} face(s)")

    summary = " | ".join(summary_parts) if summary_parts else "No match found"

    return {"results": results, "message": summary}