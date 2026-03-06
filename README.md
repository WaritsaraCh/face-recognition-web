# Real-time Face Recognition Web App

A modern web application for real-time face detection and recognition. This project allows users to register new faces into a database and perform real-time recognition using a web camera.

## Features

* **Real-time Recognition:** Detect and recognize faces instantly using a live webcam feed via WebSocket.
* **Face Registration:** Upload new face images to the database using a simple drag-and-drop interface.
* **Live Results:** Display bounding boxes, recognized names, and confidence scores directly on the video stream.
* **Database Management:** View the number of registered profiles and reload the database directly from the web interface.

## Tech Stack

* **Frontend:** HTML5, CSS3 (Glassmorphism UI), Vanilla JavaScript.
* **Backend:** Python, FastAPI, Uvicorn.
* **AI / Computer Vision:** InsightFace, ONNX Runtime.

## Prerequisites

* Python 3.11 or higher.
* A working webcam.
* (Optional but recommended) NVIDIA GPU with CUDA 12.x and cuDNN installed for hardware acceleration.
