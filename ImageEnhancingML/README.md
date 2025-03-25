# Image Enhancing using Machine Learning and OpenCV

This project uses Machine Learning (Gradient Boosting Regressor) and OpenCV to enhance image quality based on various image parameters such as Brightness, Contrast, Exposure, Shadow, and Tint.

## 🔧 Features

- Auto crop and add white border around subjects
- Extract image quality parameters
- Predict enhancements using trained ML models
- Auto-adjust image based on predicted parameters

## 📁 Folder Structure

ImageEnhancingML/
│
├── data/
│   ├── brightness.csv
│   ├── contrast.csv
│   ├── exposure.csv
│   ├── shadow.csv
│   └── tint.csv
│
├── images/
│   ├── input/                   # Before_images
│   └── output/                  # After_images
│
├── model_files/
│   ├── deploy.prototxt
│   └── mobilenet_iter_73000.caffemodel
│
├── main.py                      
├── requirements.txt             
└── README.md                    
