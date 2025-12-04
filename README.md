Malaria Cell Image Classification using MobileNetV2

A high-accuracy deep learning system for detecting Parasitized and Uninfected malaria-infected blood smear images using TensorFlow, MobileNetV2, and Transfer Learning.
This project loads the Malaria dataset from TensorFlow Datasets (TFDS), preprocesses it, trains a CNN model, fine-tunes it, evaluates performance, and generates visualizations such as confusion matrix, ROC curve, and prediction samples.

<div align="center">








</div>



## 🚀 Features

-	Automated dataset loading from TFDS: malaria dataset
-	Balanced image augmentation (flip, brightness, contrast)
-	Transfer Learning using MobileNetV2 (ImageNet weights)
-	Two-stage optimized training:
    -	Stage 1 – Train top layers
    -	Stage 2 – Fine-tune deeper layers
-	Generates:
    -	Confusion Matrix
    -	ROC Curve
    -	Accuracy/Loss graphs
    -	Prediction visualization
-	Auto-saves:
    -	best_model.h5
    -	malaria_detector_final.h5
-	Fully compatible with Google Colab and local GPU


📂 Project Structure

```
Malaria-Diagnosis/
│
├── malaria_diagnosis.py          # Main training + evaluation pipeline
├── models/                       # Contains saved models (auto-created)
│   ├── best_model.h5
│   └── malaria_detector_final.h5
│
├── visualizations/               # Generated graphs and outputs (auto-created)
│   ├── samples.png
│   ├── training_history.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── predictions.png
│
└── README.md

```

## 🛠️ Requirements
Install dependencies:
```
pip install tensorflow tensorflow-datasets numpy matplotlib seaborn scikit-learn
```
Recommended:
-	GPU support (Colab / CUDA)
-	Python 3.8+


## 🧪 How to Run the Project

Step 1: Run the Script
Execute:
```
python malaria_diagnosis.py
```
The script will automatically:
-	Load & preprocess the dataset
-	Split into train/val/test
-	Build and train MobileNetV2
-	Fine-tune deeper layers
-	Save best/final models
-	Generate visualizations

📘 What Happens Inside the Script?
1. Data Pipeline
-	Dataset: 27,558 images
-	Splits:
    -	80% Training
    -	10% Validation
    -	10% Testing
2. Model Architecture
-	MobileNetV2 (ImageNet pretrained)
-	Added classification head:
    -	GlobalAveragePooling
    -	Dense 256 → BN → Dropout(0.5)
    -	Dense 128 → BN → Dropout(0.3)
    -	Dense 1 (Sigmoid)
3. Training Strategy
-	Stage 1: Train only top layers
-	Stage 2: Unfreeze last ~50 layers and fine-tune
-	Uses callbacks:
    -	ModelCheckpoint
    -	EarlyStopping
    -	ReduceLROnPlateau
4. Evaluation Metrics
-	Accuracy
-	Precision
-	Recall
-	AUC
-	Confusion Matrix
-	ROC Curve

## 📊 Output Visualizations
Generated automatically in ```visualizations/```:
-	samples.png – 9 sample images
-	training_history.png – accuracy, loss, precision, recall graphs
-	confusion_matrix.png
-	roc_curve.png
-	predictions.png – predicted vs actual labels

## 🔍 Prediction After Training
Load the trained model and classify new images:
```
model = tf.keras.models.load_model('models/malaria_detector_final.h5')

def predict(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0][0]

    if pred < 0.5:
        return f"Parasitized ({(1 - pred) * 100:.1f}% confidence)"
    else:
        return f"Uninfected ({pred * 100:.1f}% confidence)"
```

## 📜 About

This project uses deep learning to automate malaria diagnosis using thin blood smear images.
-	Dataset: TensorFlow Datasets – Malaria
-	Model: MobileNetV2 with fine-tuning
-	Goal: Support medical diagnosis with fast, reliable predictions

## ⭐ Resources

-	TensorFlow Datasets
-	MobileNetV2 Research Paper
-	Keras API Documentation





