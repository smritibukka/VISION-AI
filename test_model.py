import tensorflow_hub as hub

print("Loading model...")

model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

print("Model loaded successfully!")