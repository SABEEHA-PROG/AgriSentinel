from pathlib import Path
import json
import sys

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "agrisentinel_rice_efficientnet.pth"
)

CLASSES_PATH = (
    PROJECT_DIR
    / "models"
    / "classes.json"
)


# ============================================================
# CHECK IMAGE ARGUMENT
# ============================================================

if len(sys.argv) < 2:

    print("❌ Please provide an image path.")

    print("\nExample:")

    print(
        r'python predict.py "C:\path\to\leaf.jpg"'
    )

    sys.exit()


IMAGE_PATH = Path(sys.argv[1])


if not IMAGE_PATH.exists():

    print("\n❌ Image not found:")
    print(IMAGE_PATH)

    sys.exit()


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("🌾 AgriSentinel Rice Disease Prediction")
print("=" * 60)

print(f"\nDevice: {DEVICE}")


# ============================================================
# LOAD CLASSES
# ============================================================

with open(
    CLASSES_PATH,
    "r",
    encoding="utf-8"
) as f:

    classes = json.load(f)


print("\nClasses:")

for i, class_name in enumerate(classes):

    print(f"{i}: {class_name}")


# ============================================================
# CREATE MODEL
# ============================================================

print("\nLoading EfficientNet-B0...")

model = models.efficientnet_b0(
    weights=None
)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    len(classes)
)


# ============================================================
# LOAD TRAINED WEIGHTS
# ============================================================

state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(state_dict)

model = model.to(DEVICE)

model.eval()


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# LOAD IMAGE
# ============================================================

print("\nLoading image:")

print(IMAGE_PATH)

image = Image.open(
    IMAGE_PATH
).convert("RGB")


image_tensor = transform(
    image
).unsqueeze(0)

image_tensor = image_tensor.to(DEVICE)


# ============================================================
# PREDICTION
# ============================================================

with torch.no_grad():

    outputs = model(
        image_tensor
    )

    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    confidence, predicted_index = torch.max(
        probabilities,
        dim=1
    )


predicted_class = classes[
    predicted_index.item()
]

confidence_value = (
    confidence.item() * 100
)


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n" + "=" * 60)

print("🌾 AGRISENTINEL RESULT")

print("=" * 60)

print(
    f"\nPrediction : {predicted_class}"
)

print(
    f"Confidence : {confidence_value:.2f}%"
)


# ============================================================
# TOP 3 PREDICTIONS
# ============================================================

print("\nTop Predictions:")

top_probabilities, top_indices = torch.topk(
    probabilities[0],
    min(3, len(classes))
)


for probability, index in zip(
    top_probabilities,
    top_indices
):

    print(
        f"{classes[index.item()]} : "
        f"{probability.item() * 100:.2f}%"
    )


print("\n" + "=" * 60)