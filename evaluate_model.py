import os
import cv2
import numpy as np
import face_recognition

# ─────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────
DATASET_DIR = "test_dataset"   # folder with test images
THRESHOLD   = 0.5              # ✅ updated to match your recognize_face.py

# ─────────────────────────────────────────
# LOAD REGISTERED EMPLOYEES FROM SUPABASE
# ─────────────────────────────────────────
from app.database import supabase

employees = supabase.table("employees").select("id,name,embedding").execute().data

known_encodings = []
known_names     = []

for emp in employees:
    enc = np.array(emp["embedding"], dtype=np.float64)
    if enc.shape[0] == 128:
        known_encodings.append(enc)
        known_names.append(emp["name"])

print(f"✅ Loaded {len(known_names)} registered employees: {known_names}")

if not known_encodings:
    print("❌ No employees found in database. Register employees first.")
    exit()

# ✅ Fix: stack into 2D array AFTER the loop (not inside)
known_encodings_np = np.stack(known_encodings)

# ─────────────────────────────────────────
# METRIC COUNTERS
# ─────────────────────────────────────────
TP = FP = TN = FN = 0
total_images   = 0
failed_detect  = 0

per_person = {}   # { name: {TP, FP, FN} } for per-person breakdown

# ─────────────────────────────────────────
# EVALUATE TEST DATASET
# ─────────────────────────────────────────
# Expected folder structure:
#   test_dataset/
#       Hanna Maria Benny/   ← folder name MUST match employee name in DB
#           img1.jpg
#           img2.jpg
#       Ann Mary Sony/
#           img1.jpg
#       Unknown/             ← optional: photos of people NOT in DB
#           img1.jpg

print(f"\n📁 Scanning test_dataset folder...\n")

for person_name in os.listdir(DATASET_DIR):
    person_path = os.path.join(DATASET_DIR, person_name)
    if not os.path.isdir(person_path):
        continue

    images = [f for f in os.listdir(person_path) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    print(f"  👤 {person_name}: {len(images)} test images")

    if person_name not in per_person:
        per_person[person_name] = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}

    for img_name in images:
        img_path = os.path.join(person_path, img_name)
        image = cv2.imread(img_path)

        if image is None:
            print(f"    ⚠️  Could not read {img_name}, skipping")
            continue

        total_images += 1
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)

        locations = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, known_face_locations=locations)

        # ✅ No face detected in test image
        if not encodings:
            failed_detect += 1
            if person_name in known_names:
                FN += 1
                per_person[person_name]["FN"] += 1
            else:
                TN += 1
                per_person[person_name]["TN"] += 1
            continue

        unknown = np.array(encodings[0], dtype=np.float64)

        # ✅ Find best match using distance
        distances    = np.linalg.norm(known_encodings_np - unknown, axis=1)
        best_idx     = int(np.argmin(distances))
        best_distance = distances[best_idx]

        predicted_name = known_names[best_idx] if best_distance < THRESHOLD else "Unknown"

        # ─────────────────────────────────
        # METRIC LOGIC
        # ─────────────────────────────────
        if person_name in known_names:
            # This is a registered employee — genuine attempt
            if predicted_name == person_name:
                TP += 1
                per_person[person_name]["TP"] += 1
            elif predicted_name == "Unknown":
                FN += 1                          # missed — should have matched
                per_person[person_name]["FN"] += 1
            else:
                FP += 1                          # wrong person identified
                per_person[person_name]["FP"] += 1
        else:
            # This is an unknown/impostor — should NOT match anyone
            if predicted_name == "Unknown":
                TN += 1                          # correctly rejected
                per_person[person_name]["TN"] += 1
            else:
                FP += 1                          # wrongly identified as someone
                per_person[person_name]["FP"] += 1

# ─────────────────────────────────────────
# CALCULATE METRICS
# ─────────────────────────────────────────
total = TP + TN + FP + FN

accuracy  = (TP + TN) / total           if total           else 0
precision = TP / (TP + FP)              if (TP + FP)       else 0
recall    = TP / (TP + FN)              if (TP + FN)       else 0
f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0
FAR       = FP / (FP + TN)             if (FP + TN)       else 0  # False Accept Rate
FRR       = FN / (FN + TP)             if (FN + TP)       else 0  # False Reject Rate

# ─────────────────────────────────────────
# PRINT RESULTS
# ─────────────────────────────────────────
print("\n" + "="*45)
print("        EVALUATION RESULTS")
print("="*45)
print(f"  Total test images   : {total_images}")
print(f"  Face detection fails: {failed_detect}")
print(f"  Threshold used      : {THRESHOLD}")
print("-"*45)
print(f"  True  Positives (TP): {TP}   ← correct match")
print(f"  True  Negatives (TN): {TN}   ← correctly rejected")
print(f"  False Positives (FP): {FP}   ← wrong person matched")
print(f"  False Negatives (FN): {FN}   ← missed / not detected")
print("-"*45)
print(f"  Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"  Precision : {precision:.4f}  ({precision*100:.2f}%)")
print(f"  Recall    : {recall:.4f}  ({recall*100:.2f}%)")
print(f"  F1 Score  : {f1:.4f}  ({f1*100:.2f}%)")
print("-"*45)
print(f"  FAR (False Accept Rate): {FAR:.4f}  ({FAR*100:.2f}%)")
print(f"  FRR (False Reject Rate): {FRR:.4f}  ({FRR*100:.2f}%)")
print("="*45)

# ─────────────────────────────────────────
# PER-PERSON BREAKDOWN
# ─────────────────────────────────────────
print("\n📊 Per-Person Breakdown:")
print(f"  {'Name':<25} {'TP':>4} {'FP':>4} {'FN':>4} {'Precision':>10} {'Recall':>8} {'F1':>8}")
print("  " + "-"*70)

for name, m in per_person.items():
    p = m["TP"] / (m["TP"] + m["FP"]) if (m["TP"] + m["FP"]) else 0
    r = m["TP"] / (m["TP"] + m["FN"]) if (m["TP"] + m["FN"]) else 0
    f = (2*p*r)/(p+r) if (p+r) else 0
    print(f"  {name:<25} {m['TP']:>4} {m['FP']:>4} {m['FN']:>4} {p*100:>9.1f}% {r*100:>7.1f}% {f*100:>7.1f}%")

print("\n✅ Evaluation complete!")
print("\n💡 Tip: Aim for F1 Score > 85% for a good final year project result.")
print("   If scores are low, try: lower threshold, more training photos,")
print("   or better lighting in test images.\n")
