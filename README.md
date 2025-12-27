# Coca Plant Detection & Custom Euclidean Tracking

This repository contains a specialized computer vision pipeline for the detection and tracking of coca plants in video streams. By leveraging **YOLOv4-tiny** for speed and a **Custom Euclidean Distance Tracker** for object persistence, this system provides an efficient alternative to heavy tracking frameworks.

## 📌 Project Overview
The workflow for this project involved:
* **Data Sourcing**: Extracted raw plant datasets from **Hugging Face**.
* **Data Management**: Used **Roboflow** for high-quality data annotation and version control.
* **Object Detection**: Implemented **YOLOv4-tiny** via the Darknet framework, optimized for 3 specific plant classes.
* **Object Tracking**: Developed a **Custom Euclidean Distance Tracker** to maintain unique IDs for plants as they move through the frame.

## 🛠️ Tech Stack
* **Neural Network**: YOLOv4-tiny (Darknet)
* **Tracking**: Custom Python Implementation (Centroid-based)
* **Data Platform**: Roboflow & Hugging Face
* **Libraries**: OpenCV, NumPy, TensorFlow
* **Infrastructure**: Google Colab (GPU accelerated)

## 📂 Project Structure
* `darknet/` - The C-based YOLO implementation.
* `tracker.py` - Custom class for Euclidean distance calculation and ID assignment.
* `yolov4-tiny-custom.cfg` - Customized configuration file for plant detection.
* `obj.names` - Class labels: `coca_plant`, `coca_plant2`, `coca_plant3`.

## ⚙️ How It Works

### 1. Detection (YOLOv4-tiny)
The model identifies the location of plants and returns bounding box coordinates $(x, y, w, h)$. YOLOv4-tiny was chosen for its balance between accuracy and real-time inference speed on edge devices.

### 2. Custom Tracking (Euclidean Distance)
Instead of using external libraries like DeepSORT, this project utilizes a custom `EuclideanDistTracker` class. 
- It calculates the **centroid** (center point) of each detection.
- It compares the distance between current centroids and previous centroids.
- If the distance is within a specific threshold, it assigns the **same ID** to the object.

```python
# Core logic of the custom tracker
distance = math.hypot(cx - pt[0], cy - pt[1])
if distance < 25:
    self.center_points[id] = (cx, cy)
    # Maintain ID persistence...
```

## Dataset Access

The annotated dataset is managed and versioned via **Roboflow**. You can pull the data directly into your environment using the following Python snippet:

```python
from roboflow import Roboflow

# Initialize Roboflow with your API Key
rf = Roboflow(api_key="YOUR_API_KEY")

# Access the specific workspace and project
project = rf.workspace("vinati").project("yolov4s")

# Download the dataset in Darknet format for YOLOv4-tiny
dataset = project.version(1).download("darknet")
```

### 💡 Note:
Ensure you replace `"YOUR_API_KEY"` with your actual private key found in your Roboflow account settings to successfully authenticate the download.

## 📊 Training Specifications

The model was trained with the following parameters to balance speed and accuracy for real-time plant tracking:

* **Input Resolution**: $416 \times 416$
* **Training Iterations**: 6,000 batches
* **Detection Classes**: 3 Classes (`coca_plant`, `coca_plant2`, `coca_plant3`)
* **Learning Schedule**: Steps at 4,800 and 5,400 iterations
* **Hardware**: Tesla T4 GPU (via Google Colab)

## 📝 Credits

This project leverages several open-source tools and platforms:

* **Dataset Management**: [Roboflow](https://roboflow.com) — Used for data annotation, version control, and preprocessing.
* **Base Model**: [Darknet YOLOv4](https://github.com/AlexeyAB/darknet) — Implementation of the YOLOv4-tiny architecture.
* **Data Source**: Hugging Face — Original source of the plant imagery dataset.
* **Tracking Logic**: Custom implementation of a Centroid-based Euclidean Distance Tracker.

