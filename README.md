# 🚗👤 Vision-Based Object Detection via Segmentation and Classification  
### A Two-Stage Deep Learning Pipeline Using the KITTI Dataset

This repository presents a **professional computer vision study** that explores an **alternative object detection strategy** based on **segmentation-first perception followed by classification**, rather than using end-to-end object detectors directly.

The project focuses on **detecting humans and cars** in road scenes by combining **semantic segmentation, blob analysis, bounding box extraction, and deep learning–based classification**.  
All experiments, preprocessing steps, models, and results are implemented in **Python** and documented through **Jupyter notebooks and a detailed technical report**.

---

## 🎯 Project Goal

The primary objective of this work is to **design and implement an object detection application** that:

- Uses **segmentation and classification deep learning models**
- Detects **human and car objects** in real-world road images
- Reduces computational cost compared to traditional object detection pipelines
- Demonstrates a **modular and interpretable vision-based detection approach**

Instead of directly predicting bounding boxes and class labels from raw images, the proposed method follows a **multi-stage perception pipeline** that separates concerns and improves interpretability.

---

## 🧠 Motivation & Problem Definition

Object detection is one of the most critical and computationally expensive problems in computer vision.  
Conventional approaches attempt to directly infer object locations and classes from an input image, which can be costly in terms of **inference time and model complexity**.

In this test project, a **different paradigm** is explored:

1. **Segment objects first**
2. **Extract connected components (blobs)**
3. **Generate bounding boxes from segmented regions**
4. **Classify each candidate region independently**

This approach enables:
- Reduced search space for classification
- Clear separation between localization and recognition
- Easier debugging and visualization of intermediate results

---

## 📂 Dataset: KITTI Vision Benchmark

The project uses the **KITTI dataset**, which contains real-world images captured from vehicles driving on roads, along with **pixel-level annotations** for multiple object classes.

From the dataset:
- Only **Human** and **Car** classes are selected
- Segmentation masks are used as ground truth
- Images are preprocessed and converted into formats suitable for training segmentation and classification models

---

## 🧩 System Architecture

The complete object detection system follows the pipeline below:

**Input RGB Image**  
→ Semantic Segmentation  
→ Blob / Connected Component Analysis  
→ Bounding Box Extraction  
→ Cropped Region Classification  
→ Final Object Detection Output  

This modular design allows each stage to be evaluated and improved independently.

---

## 🧪 Implemented Components

### 1️⃣ Data Preparation & YOLO-Compatible Formatting

The notebook  
**`rgb_to_grayscale_and_mask_to_yolo_format.ipynb`**  
is responsible for:

- Converting RGB images to grayscale when needed
- Processing segmentation masks
- Generating **YOLO-style annotations** from segmentation data
- Preparing clean and structured datasets for training

This step ensures consistency between segmentation outputs and bounding box–based learning workflows.

---

### 2️⃣ Segmentation & Classification Pipeline

The notebook  
**`artificial_vision_test_project.ipynb`**  
contains the **core implementation of the project**, including:

- Training and evaluation of **segmentation models**
- Extraction of segmented blobs
- Bounding box generation from connected components
- Training and testing of **classification models**
- End-to-end object detection logic
- Quantitative and qualitative evaluation

Both segmentation and classification models are implemented using **deep learning techniques** and evaluated on KITTI images.

---

## 3️⃣ Visual Results & Explanations

The following figures correspond directly to the report and illustrate key stages of the pipeline:

- **Figure 1:** Original KITTI images and corresponding ground-truth segmentation masks
  
  ![Figure 3](images/image3.png)

- **Figure 2:** Segmentation model outputs and blob extraction results
  
  ![Figure 5](images/image5.png)  


- **Figure 3–4:** Bounding box generation from segmented regions

  ![Figure 7](images/image7.png)  
  ![Figure 8](images/image8.png)  

- **Figure 5:** Final object detection results after classification  

  ![Figure 11](images/image11.png)

These visualizations clearly demonstrate how segmentation-driven detection works step by step.

---

## ⚙️ Technologies & Tools

- **Programming Language:** Python  
- **Environment:** Jupyter Notebook  
- **Deep Learning:** CNN-based segmentation and classification models  
- **Computer Vision:** OpenCV  
- **Dataset:** KITTI Vision Benchmark  
- **Annotation Format:** YOLO-compatible bounding boxes  

---

## 🚀 Key Contributions

✔ Segmentation-first object detection strategy  
✔ Blob-based bounding box generation  
✔ Human & car detection on real-world road scenes  
✔ Clear separation of segmentation and classification stages  
✔ Fully reproducible notebooks and documented pipeline  

---

## 🔬 Conclusion

This project demonstrates that **object detection can be effectively achieved by combining segmentation and classification models**, rather than relying solely on monolithic detectors.

The proposed approach:
- Improves interpretability of detection results
- Enables modular development and testing
- Serves as a strong educational and experimental reference for vision-based perception systems

The methodology is particularly suitable for:
- Autonomous driving perception
- Robotics applications
- Embedded and resource-constrained systems
- Academic research and teaching purposes

---

## 📬 Contact

**Furkan Karakaya**  
AI & Computer Vision Engineer  
📧 se.furkankarakaya@gmail.com  

---

⭐ If you find this project useful, consider starring the repository!
