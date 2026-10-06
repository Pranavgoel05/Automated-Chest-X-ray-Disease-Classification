# Automated Chest X-ray Disease Classification

An automated, explainable deep learning pipeline for multi-class chest X-ray disease classification built using EfficientNet-B0. The system incorporates data preprocessing, class imbalance handling, classical machine learning comparisons, explainability via Grad-CAM, and an interactive Streamlit web deployment. 

# 📌 Features

<b> 1) Deep Learning Model :</b> EfficientNet-B0 architecture fine-tuned with a custom classification head (Global Average Pooling, 0.4 Dropout, FC layer). 

<b> 2) Class Imbalance Mitigation :</b> Implements Weighted Cross-Entropy Loss and targeted data augmentations (rotation, flip, jitter, scaling) to address heavily skewed class distributions. 

<b> 3) Visual Explainability :</b> Integrated Grad-CAM heatmaps overlaid on X-rays to highlight active regions driving predictions for medical interpretability.

<b> 4) Interactive UI :</b> A Streamlit web application enabling real-time X-ray uploads, confidence score outputs, and visual Grad-CAM diagnosis overlays.

# 📊 Dataset & Preprocessing
<b>Dataset :</b> NIH ChestXray14 dataset consisting of 112,120 chest X-ray images spanning 14 disease categories plus "No Finding". 

<b>Preprocessing Pipeline :</b> 
#### -> Resizing images to $224 \times 224$.   
#### -> Grayscale-to-RGB conversion and ImageNet normalization.   
#### -> Patient-wise train/validation/test split to prevent data leakage.  

# 📈 Performance & Results
### Classical Model Comparison (AUC Scores)
| Model      | AUC Score         
| ------------- |:-------------:|
|Random Forest (NumPy)       | 0.6946 | 
| CatBoost      | 0.6730      | 
| Naive Bayes | 0.5848      | 
| MLP (PyTorch) | 0.5169 |
|Logistic Regression (Pytorch) | 0.4730 |

## Deep Learning Model Highlights
### Training Setup: Adam Optimizer with Cosine Annealing Learning Rate Scheduler over 8 epochs. 
### Sample Diagnostic Outputs:
#### -> Edema Detection: 78.59% confidence.
#### -> Cardiomegaly Detection: 87.13% confidence. 
#### -> Emphysema Detection: 89.27% confidence.   

# 🚀 Web Application Demonstration
### The Streamlit deployment provides end-to-end inferencing with interactive visualizations : 
#### -> Upload: Drag-and-drop chest X-ray files (JPG, PNG, JPEG).
#### -> Prediction: Top predicted class with associated percentage confidence.
#### -> Interpretability: Side-by-side original image, isolated Grad-CAM heatmap, and blended overlay. 

# 🔮 Future Enhancements

### -> Integrate lung segmentation preprocessing pipeline.  
### -> Train larger EfficientNet variants (B3–B5) and ensemble models.   
### -> Apply model calibration methods to refine probability confidence.  
### -> Deploy application using ONNX Runtime / Streamlit Cloud.   
