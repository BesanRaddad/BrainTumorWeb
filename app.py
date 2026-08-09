from flask import Flask, render_template, request
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import base64
from io import BytesIO


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class_names = [
    "glioma",
    "meningioma",
    "pituitary",
    "notumor"
]


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


model = models.resnet50(weights=None)

model.fc = nn.Sequential(
    nn.Dropout(),
    nn.Linear(model.fc.in_features, 4)
)


model_path = os.path.join(
    os.path.dirname(__file__),
    "best_brain_model.pth"
)

checkpoint = torch.load(
    model_path,
    map_location=device
)

model.load_state_dict(checkpoint)
model.to(device)
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def image_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="JPEG")

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


def predict_image(image):
    image = image.convert("RGB")

    tensor = transform(image)
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)[0]

    index = torch.argmax(probabilities).item()

    prediction = class_names[index]
    confidence = probabilities[index].item() * 100

    all_probabilities = {
        name: probabilities[i].item() * 100
        for i, name in enumerate(class_names)
    }

    return prediction, confidence, all_probabilities


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
                image = Image.open(file.stream).convert("RGB")

                prediction, confidence, probabilities = predict_image(image)

                image_data = image_to_base64(image)

                info = condition_info[prediction]

            except Exception as e:
                print(e)
                error = "Unable to process this image. Please upload a valid MRI image."

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        probabilities=probabilities,
        image_data=image_data,
        info=info,
        error=error
    )


if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )