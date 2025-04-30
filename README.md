# 🌥️ Cloud Masking in Satellite Imagery Using U-Net

This project implements cloud segmentation on satellite images using a U-Net model with 4-channel input (RGB + IR). It also includes classical model trials and inference scripts, structured for smooth Kaggle or local runs.

---

## 📁 Repository Structure

```
.
├── README.md                        # Project documentation
├── ST-Project.ipynb                 # Main notebook: data prep, training, evaluation
├── ST-Project-Classical-Model.ipynb # Classical ML model 
├── run_inference.py                 # Inference script for local use
├── run_inference.ipynb              # Kaggle-ready inference notebook
├── submission.csv                   # Final/test submission file (RLE encoded)
├── model_logs.txt                   # Training logs (e.g., Dice scores, losses)
```

---

## 🧠 Model Details

- **Architecture:** U-Net
- **Input Channels:** 4 (Red, Green, Blue, Infrared)
- **Output:** Binary cloud mask (1 = cloud, 0 = clear)
- **Metric:** Dice Coefficient  
- **Best Dice Achieved:** **96.25%**

🎯 **Download Trained Model:**  
[🔗 model_best_dice_96.25.pth](https://drive.google.com/file/d/1xgNJzqIRz8NddqkU4Sts-_wpCan2anWv/view?usp=sharing)

---

## 🚀 How to Run

### ✅ On Kaggle

1. Upload test `.tif` images under:

```
/test/test/data/
    ├── 0001.tif
    ├── 0002.tif
    └── ...
```

2. Upload `sample_submission.csv`
3. Open and run `run_inference.ipynb`

---

### 🖥️ Locally

1. Install dependencies:
```bash
pip install torch torchvision numpy matplotlib pandas tifffile
```

2. Download the trained model and place it in the project directory.

3. Run:
```bash
python run_inference.py
```

- This generates `submission.csv` and visualizes cloud masks.

---

## 📌 Notes

- `ST-Project-Classical-Model.ipynb` contains an SVM model (abandoned due to long runtime).
- `model_logs.txt` tracks training metrics and progress.

---

## 👨‍💻 Author

Developed by **Team 01**  
Final project for **Satellite & Remote Sensing Course (CMP25)**  
Faculty of Engineering, Cairo University
