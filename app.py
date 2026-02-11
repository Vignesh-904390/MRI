import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, render_template, request, jsonify

# =====================
# BASIC CONFIG (MRI)
# =====================
MODEL_PATH = "model/mri_effnet_best.pth"
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🔌 Device:", DEVICE)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# =====================
# LOAD MRI DISEASE INFO JSON
# =====================
with open("mri_disease.json", "r", encoding="utf-8") as f:
    raw_disease_info = json.load(f)

# Build normalized lookup
DISEASE_INFO = {}
for orig_key, info in raw_disease_info.items():
    norm = orig_key.strip().lower()
    DISEASE_INFO[norm] = {"orig_key": orig_key, "info": info}

# =====================
# MRI LABEL MAP
# =====================
LABEL_MAP = {
    "multiple_sclerosis_lesions": {
        "category": "Multiple Sclerosis Lesions",
        "diseases": [
            "cervical spinal cord ms lesions",
            "clinically isolated syndrome (cis)",
            "cortical ms lesions",
            "juxtacortical ms lesions",
            "normal (multiple sclerosis lesions)",
            "optic nerve demyelination",
            "periventricular ms lesions",
            "radiologically isolated syndrome (ris)",
            "relapsing–remitting multiple sclerosis",
            "small vessel ischemic disease"
        ]
    },

    "pituitary_microadenoma": {
        "category": "Pituitary Microadenoma Conditions",
        "diseases": [
            "normal (pituitary microadenoma)",
            "isointense pituitary microadenoma",
            "dynamic contrast–delayed enhancing microadenoma",
            "prolactinoma (microprolactinoma)",
            "acth-secreting microadenoma",
            "gh-secreting microadenoma",
            "pituitary hyperplasia",
            "pituitary microadenoma"
        ]
    }
}

# normalize LABEL_MAP
LABEL_MAP = {
    k.lower(): {
        "category": v["category"],
        "diseases": [d.lower() for d in v["diseases"]]
    }
    for k, v in LABEL_MAP.items()
}

# =====================
# LOAD MRI MODEL
# =====================
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

class_names_raw = checkpoint.get("classes", [])
class_names = [c.strip() for c in class_names_raw]
class_names_norm = [c.strip().lower() for c in class_names_raw]

num_classes = len(class_names)

model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()

print("✅ MRI model loaded with classes:", class_names)

# =====================
# IMAGE TRANSFORM
# =====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =====================
# HELPERS
# =====================
def field_to_list(v):
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, str):
        parts = []
        for seg in v.split("\n"):
            parts.extend(seg.split(";"))
        final = []
        for seg in ",".join(parts).split(","):
            item = seg.strip()
            if item:
                final.append(item)
        return final
    return [str(v)]

def find_info_for_label(raw_label):
    if not raw_label:
        return None, None

    label_norm = raw_label.strip().lower()

    if label_norm in DISEASE_INFO:
        rec = DISEASE_INFO[label_norm]
        return rec["orig_key"], rec["info"]

    if label_norm in LABEL_MAP:
        for candidate in LABEL_MAP[label_norm]["diseases"]:
            if candidate in DISEASE_INFO:
                rec = DISEASE_INFO[candidate]
                return rec["orig_key"], rec["info"]

    if "(" in label_norm:
        parent = label_norm.split("(", 1)[0].strip()
        if parent in DISEASE_INFO:
            rec = DISEASE_INFO[parent]
            return rec["orig_key"], rec["info"]
        if parent in LABEL_MAP:
            for candidate in LABEL_MAP[parent]["diseases"]:
                if candidate in DISEASE_INFO:
                    rec = DISEASE_INFO[candidate]
                    return rec["orig_key"], rec["info"]

    for k, rec in DISEASE_INFO.items():
        if label_norm in k or k in label_norm:
            return rec["orig_key"], rec["info"]

    return None, None

# =====================
# ROUTES
# =====================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"})

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"})

        save_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(save_path)

        img = Image.open(save_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, 1)

        raw_label = (
            class_names[pred_idx.item()]
            if pred_idx.item() < len(class_names)
            else str(pred_idx.item())
        )

        confidence = float(conf.item()) * 100.0

        display_name, info = find_info_for_label(raw_label)

        if info is None:
            info = {
                "Description": f"Information for '{raw_label}' is being updated.",
                "Symptoms": "",
                "Diagnosis": "",
                "Medicines": "",
                "Prevention": "",
                "Care": ""
            }
            display_name = raw_label

        response_info = {
            "Description": info.get("Description", ""),
            "Symptoms": field_to_list(info.get("Symptoms", "")),
            "Diagnosis": field_to_list(info.get("Diagnosis", "")),
            "Medicines": field_to_list(info.get("Medicines", "")),
            "Prevention": field_to_list(info.get("Prevention", "")),
            "Care": field_to_list(info.get("Care", ""))
        }

        return jsonify({
            "disease": display_name,
            "model_label": raw_label,
            "confidence": round(confidence, 2),
            "image": file.filename,
            "info": response_info
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =====================
# RUN
# =====================
if __name__ == "__main__":
    app.run(debug=True)
