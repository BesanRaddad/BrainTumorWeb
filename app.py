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

# Render Free has limited CPU/RAM.
# Using fewer threads helps reduce memory usage.

torch.set_num_threads(1)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


# =========================================================
# Device
# =========================================================

# Render uses CPU
device = torch.device("cpu")

print("Using device:", device)


# =========================================================
# Class names - Brain Tumor Model
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
# Paths
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

brain_model_path = os.path.join(
    BASE_DIR,
    "best_brain_model.pth"
)

mri_validator_path = os.path.join(
    BASE_DIR,
    "best_mri_validator.pth"
)


# =========================================================
# =========================================================
# MRI VALIDATOR - MobileNetV3-Small
# =========================================================
# =========================================================

print("Loading MRI validator...")

mri_validator = models.mobilenet_v3_small(
    weights=None
)

# The classifier was changed during training
# from the original output to 2 classes:
#
# 0 = Not Brain MRI
# 1 = Brain MRI

validator_in_features = (
    mri_validator.classifier[3].in_features
)

mri_validator.classifier[3] = nn.Linear(
    validator_in_features,
    2
)


# Load trained validator weights
validator_checkpoint = torch.load(
    mri_validator_path,
    map_location="cpu"
)

mri_validator.load_state_dict(
    validator_checkpoint
)

# Release checkpoint memory
del validator_checkpoint

mri_validator.to(device)

mri_validator.eval()


# Threshold selected from validation analysis
MRI_THRESHOLD = 0.80


print("MRI validator loaded successfully.")
print("MRI validation threshold:", MRI_THRESHOLD)


# =========================================================
# Brain MRI Validator Transform
# =========================================================

validator_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# =========================================================
# BRAIN TUMOR MODEL - ResNet50
# =========================================================
# =========================================================

print("Loading brain tumor model...")


model = models.resnet50(
    weights=None
)


model.fc = nn.Sequential(
    nn.Dropout(),
    nn.Linear(
        model.fc.in_features,
        4
    )
)


# =========================================================
# Load Brain Tumor Model
# =========================================================

checkpoint = torch.load(
    brain_model_path,
    map_location="cpu"
)

model.load_state_dict(
    checkpoint
)

# Release checkpoint memory
del checkpoint

model.to(device)

model.eval()


print("Brain tumor model loaded successfully.")


# =========================================================
# Brain Tumor Image Transformation
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

    # Resize image for website display
    display_image.thumbnail(
        (800, 800)
    )

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
# MRI VALIDATION
# =========================================================

def validate_brain_mri(image):

    """
    Returns:

    brain_probability:
        Probability that the image is Brain MRI.

    is_brain_mri:
        True  -> Brain MRI
        False -> Not Brain MRI
    """

    image = ImageOps.exif_transpose(
        image
    )

    image = image.convert("RGB")

    # Apply the SAME preprocessing used during
    # validator training.
    tensor = validator_transform(
        image
    )

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(device)


    with torch.inference_mode():

        outputs = mri_validator(
            tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]

        # Class 1 = Brain MRI
        brain_probability = (
            probabilities[1].item()
        )


    is_brain_mri = (
        brain_probability >= MRI_THRESHOLD
    )


    # Release tensors
    del tensor
    del outputs
    del probabilities


    return (
        is_brain_mri,
        brain_probability
    )


# =========================================================
# BRAIN TUMOR PREDICTION
# =========================================================

def predict_image(image):

    image = ImageOps.exif_transpose(
        image
    )

    image = image.convert("RGB")


    # Resize only for model inference
    tensor = transform(
        image
    )

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(device)


    with torch.inference_mode():

        output = model(
            tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )[0]


    index = torch.argmax(
        probabilities
    ).item()


    prediction = class_names[
        index
    ]


    confidence = (
        probabilities[index].item()
        * 100
    )


    all_probabilities = {

        name: probabilities[i].item() * 100

        for i, name in enumerate(
            class_names
        )
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
# Main Route
# =========================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
def index():

    prediction = None

    confidence = None

    probabilities = None

    image_data = None

    info = None

    error = None


    if request.method == "POST":

        file = request.files.get(
            "mri_image"
        )


        # =================================================
        # Check file
        # =================================================

        if not file or file.filename == "":

            error = (
                "Please select an MRI image."
            )


        else:

            try:

                # ==========================================
                # Open uploaded image
                # ==========================================

                image = Image.open(
                    file.stream
                )


                # ==========================================
                # Fix orientation
                # ==========================================

                image = ImageOps.exif_transpose(
                    image
                )


                # ==========================================
                # Convert to RGB
                # ==========================================

                image = image.convert(
                    "RGB"
                )


                # ==========================================
                # Limit extremely large images
                # ==========================================

                max_dimension = 2000


                if (
                    image.width > max_dimension
                    or image.height > max_dimension
                ):

                    image.thumbnail(
                        (
                            max_dimension,
                            max_dimension
                        ),
                        Image.Resampling.LANCZOS
                    )


                # ==========================================
                # Create image preview
                # ==========================================

                image_data = image_to_base64(
                    image
                )


                # =================================================
                # STEP 1
                # Validate whether image is Brain MRI
                # =================================================

                (
                    is_brain_mri,
                    brain_probability
                ) = validate_brain_mri(
                    image
                )


                print(
                    "Brain MRI probability:",
                    f"{brain_probability * 100:.2f}%"
                )


                # =================================================
                # STEP 2
                # Reject if NOT Brain MRI
                # =================================================

                if not is_brain_mri:

                    error = (
                        "Uploaded image is not a brain MRI."
                    )

                    print(
                        "Validation result: "
                        "NOT BRAIN MRI"
                    )


                # =================================================
                # STEP 3
                # Only Brain MRI reaches ResNet50
                # =================================================

                else:

                    print(
                        "Validation result: "
                        "BRAIN MRI"
                    )


                    (
                        prediction,
                        confidence,
                        probabilities
                    ) = predict_image(
                        image
                    )


                    info = condition_info.get(
                        prediction
                    )


            except Exception as e:

                print(
                    "IMAGE PROCESSING ERROR:"
                )

                print(e)


                error = (
                    "Unable to process this image. "
                    "Please upload a valid image."
                )


    # =========================================================
    # Render website
    # =========================================================

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
        os.environ.get(
            "PORT",
            5000
        )
    )


    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )