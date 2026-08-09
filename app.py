from flask import Flask, render_template, request
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageOps
import os
import base64
from io import BytesIO


# =========================================================
# Flask App
# =========================================================

app = Flask(__name__)

# Maximum upload size: 5 MB
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024


# =========================================================
# CPU optimization
# =========================================================

# Render Free has very limited CPU/RAM.
# Using fewer threads helps prevent memory spikes.
torch.set_num_threads(1)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


device = torch.device("cpu")


# =========================================================
# Class names
# =========================================================

class_names = [
    "glioma",
    "meningioma",
    "pituitary",
    "notumor"
]


# =========================================================
# Condition information
# =========================================================

condition_info = {
    "glioma": {
        "title": "Glioma",
        "description": (
            "Glioma is a type of tumor that develops from glial cells "
            "in the brain or spinal cord. Gliomas can vary in type and "
            "severity, so further medical evaluation is required to "
            "understand the specific condition."
        )
    },

    "meningioma": {
        "title": "Meningioma",
        "description": (
            "Meningioma is a tumor that develops from the meninges, "
            "the protective membranes surrounding the brain and spinal cord. "
            "Many meningiomas grow slowly, but their significance depends "
            "on their size, location, and other clinical factors."
        )
    },

    "pituitary": {
        "title": "Pituitary Tumor",
        "description": (
            "A pituitary tumor develops in or around the pituitary gland, "
            "a small gland located at the base of the brain. Some pituitary "
            "tumors may affect hormone production and require medical evaluation."
        )
    },

    "notumor": {
        "title": "No Tumor Detected",
        "description": (
            "The model classified this MRI image as belonging to the "
            "No Tumor category. This means that the image was most similar "
            "to the No Tumor examples learned by the model. This result "
            "does not rule out other medical conditions."
        )
    }
}


# =========================================================
# Build model
# =========================================================

print("Loading brain tumor model...")


model = models.resnet50(weights=None)

model.fc = nn.Sequential(
    nn.Dropout(),
    nn.Linear(model.fc.in_features, 4)
)


# =========================================================
# Model path
# =========================================================

model_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "best_brain_model.pth"
)


# =========================================================
# Load model
# =========================================================

checkpoint = torch.load(
    model_path,
    map_location="cpu"
)

model.load_state_dict(checkpoint)

# Release checkpoint memory after loading
del checkpoint

model.to(device)
model.eval()

print("Brain tumor model loaded successfully.")


# =========================================================
# Image transformation
# =========================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# Convert image to Base64
# =========================================================

def image_to_base64(image):

    # Make a copy so we don't modify the original image
    display_image = image.copy()

    # Resize the image used for displaying on the website.
    # This prevents sending a huge image back to the browser.
    display_image.thumbnail((800, 800))

    buffer = BytesIO()

    display_image.save(
        buffer,
        format="JPEG",
        quality=80,
        optimize=True
    )

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


# =========================================================
# Prediction
# =========================================================

def predict_image(image):

    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    # Resize only for model inference
    tensor = transform(image)

    tensor = tensor.unsqueeze(0)

    # CPU
    tensor = tensor.to(device)

    # inference_mode uses less memory than normal inference
    with torch.inference_mode():

        output = model(tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )[0]

    index = torch.argmax(probabilities).item()

    prediction = class_names[index]

    confidence = (
        probabilities[index].item() * 100
    )

    all_probabilities = {
        name: probabilities[i].item() * 100
        for i, name in enumerate(class_names)
    }

    # Release tensors
    del tensor
    del output
    del probabilities

    return (
        prediction,
        confidence,
        all_probabilities
    )


# =========================================================
# Main route
# =========================================================

@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    confidence = None
    probabilities = None
    image_data = None
    info = None
    error = None

    if request.method == "POST":

        file = request.files.get("mri_image")

        if not file or file.filename == "":
            error = "Please select an MRI image."

        else:

            try:

                # Open uploaded image
                image = Image.open(file.stream)

                # Convert to RGB
                image = ImageOps.exif_transpose(image)
                image = image.convert("RGB")

                # Limit extremely large images
                # before processing.
                max_dimension = 2000

                if (
                    image.width > max_dimension
                    or image.height > max_dimension
                ):

                    image.thumbnail(
                        (max_dimension, max_dimension),
                        Image.Resampling.LANCZOS
                    )

                # Run AI prediction
                (
                    prediction,
                    confidence,
                    probabilities
                ) = predict_image(image)

                # Create smaller image for website
                image_data = image_to_base64(image)

                # Get information about prediction
                info = condition_info.get(prediction)

            except Exception as e:

                print("IMAGE PROCESSING ERROR:")
                print(e)

                error = (
                    "Unable to process this image. "
                    "Please upload a valid MRI image."
                )

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        probabilities=probabilities,
        image_data=image_data,
        info=info,
        error=error
    )


# =========================================================
# Local development
# =========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5000)
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )