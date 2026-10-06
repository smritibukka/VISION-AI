import tensorflow_hub as hub
import numpy as np
from labels import COCO_LABELS

MODEL_URL = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"

print("Loading object detection model...")
model = hub.load(MODEL_URL)
print("Object detection model loaded!")


def detect_objects(image, threshold=0.5):
    # Convert image to NumPy array
    image = np.array(image)

    # Add batch dimension
    input_tensor = np.expand_dims(image, axis=0)

    # Run the image through the AI model
    results = model(input_tensor)

    # Get detection information
    boxes = results["detection_boxes"][0].numpy()
    classes = results["detection_classes"][0].numpy().astype(int)
    scores = results["detection_scores"][0].numpy()

    detections = []

    for box, class_id, score in zip(boxes, classes, scores):

        if score >= threshold:
            # Convert class ID into object name
            object_name = COCO_LABELS.get(class_id, "unknown")

            detections.append({
                "name": object_name,
                "class_id": class_id,
                "score": float(score),
                "box": box.tolist()
            })

    return detections