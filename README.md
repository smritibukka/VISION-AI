# 👁️ VISION-AI — AI Object Detection System

VISION-AI is an AI-powered object detection application that analyzes images and identifies objects present in them.

The application uses a pre-trained SSD MobileNet V2 model with TensorFlow Hub to detect objects, estimate confidence scores, count detected objects, and draw bounding boxes around them.

---

## 📌 Problem Statement

Identifying multiple objects in an image manually can be time-consuming and inefficient.

VISION-AI provides an automated solution that can analyze an image and identify multiple objects using an AI-based object detection model.

---

## 🎯 Objective

The main objective of VISION-AI is to develop an easy-to-use application that can:

- Detect multiple objects in an image.
- Identify the names of detected objects.
- Display confidence scores.
- Draw bounding boxes around detected objects.
- Count detected objects.
- Display the number of unique object categories.
- Allow users to upload images.
- Allow users to capture images using a camera.
- Allow users to control the detection confidence threshold.
- Allow users to download detection results as a CSV file.

---

## ✨ Features

- 🖼️ Image upload
- 📷 Camera input
- 🤖 AI object detection
- 📦 Bounding boxes
- 📊 Confidence scores
- 🎚️ Adjustable confidence threshold
- 🔢 Object counting
- 📋 Detection results table
- 📥 CSV result download
- 🛡️ Error handling
- 📈 Detection statistics

---

## 🧠 AI Model

VISION-AI uses the **SSD MobileNet V2** object detection model through TensorFlow Hub.

SSD stands for **Single Shot MultiBox Detector**.

The model is pre-trained to recognize common objects.

The application uses the model to generate:

- Object class IDs
- Object confidence scores
- Bounding box coordinates

These results are then converted into human-readable object names using COCO dataset labels.

---

## 🏗️ Project Architecture

```text
User
  ↓
Streamlit UI
  ↓
Image Upload / Camera
  ↓
Image Processing
  ↓
SSD MobileNet V2
  ↓
Object Detection
  ↓
Object Name + Confidence + Bounding Box
  ↓
Confidence Threshold Filtering
  ↓
Object Counting
  ↓
Annotated Image + Results Table
  ↓
CSV Download