# Codtech-T2
**Name** : Abhinav Gurram  
**Company**: CODTECH IT SOLUTIONS PVT.LTD  
**Intern ID**: CT6WDS2146   
**Domain**: ARTIFICIAL INTELLIGENCE  
**Duration**: OCTOBER 10th, 2024 to NOVEMBER 25th, 2024  
**Mentor**: Santhosh

# CODTECH-Task2
# Task4: Computer Vision - EfficientNetB0 Model for Image Classification

## Steps in Computer Vision Task

### 1. **Model Architecture Setup**
**Base Model**: The **EfficientNetB0** model is loaded, pretrained on **ImageNet**, without its top classification layers (`include_top=False`). This model serves as the feature extractor.
- **Input Size**: The input shape for the model is set to `(224, 224, 3)` to match EfficientNetB0's default input size.
- **Freezing the Base Model**: The pretrained base model is frozen by setting `trainable = False`. This ensures the weights of the base model are not updated during training.

### 2. **Model Architecture Definition**
- **GlobalAveragePooling2D**: A pooling layer is added to reduce the spatial dimensions of the feature maps.
- **Dense Layer**: A fully connected layer with **128 units** and **ReLU** activation is added to introduce non-linearity.
- **Output Layer**: A dense layer with **30 units** (for 30 classes) and **Softmax** activation is added to output class probabilities.

### 3. **Model Compilation**
- **Optimizer**: The model is compiled using the **Adam optimizer**.
- **Loss Function**: The **categorical crossentropy** loss function is used as the problem is a multi-class classification task.
- **Metrics**: The accuracy of the model is monitored during training.

### 4. **Model Training with Callbacks**
- **Early Stopping**: The **EarlyStopping** callback is used to stop training when the validation loss stops improving for 3 consecutive epochs.
- **Learning Rate Reduction**: The **ReduceLROnPlateau** callback reduces the learning rate by a factor of 0.2 if the validation loss does not improve for 3 epochs, to allow the model to fine-tune with a lower learning rate.

The model is trained for **100 epochs**, with training and validation data passed through `train_generator` and `val_generator`.

### 5. **Training and Validation Metrics**
- **Training Accuracy** and **Validation Accuracy** are plotted to visualize the model's performance over epochs.
- **Training Loss** and **Validation Loss** are plotted to observe overfitting or underfitting.

### 6. **Model Evaluation on Test Data**
- A test generator is created using **ImageDataGenerator** with pixel value rescaling to match the model's input requirements.
- The model is then evaluated on the test dataset using `model.evaluate()`, and the **test accuracy** is printed.

### 7. **Classification Report**
- Predictions are made on the **validation set** using `model.predict()`.
- The predicted labels are compared with the true labels to generate a **classification report** using **scikit-learn's `classification_report()`**, which provides key metrics such as **precision**, **recall**, and **F1-score** for each class.

### 8. **Class Labels and Predictions on External Images**
- **External Image Prediction**: A custom external image (e.g., `'Pics/unnamed.jpg'`) is loaded, preprocessed, and passed to the model for prediction.
- The model predicts the class label for the external image and prints the **predicted class label**.

### 9. **Model Saving**
- The trained model is saved as `"efficientnet_model.h5"` for future use or deployment.

---

## **Key Features of this Task**
1. **EfficientNetB0 Model**: Utilizes the lightweight and efficient EfficientNetB0 architecture, pretrained on ImageNet, as the base model for feature extraction.
2. **Data Augmentation**: Applied to training data to improve the model's ability to generalize.
3. **Early Stopping**: Prevents overfitting by stopping training when validation loss plateaus.
4. **Learning Rate Adjustment**: Dynamically adjusts the learning rate to improve convergence during training.
5. **Classification Report**: Provides detailed evaluation metrics like precision, recall, and F1-score for each class in the dataset.
6. **External Image Prediction**: The model is used to predict classes for external images, demonstrating its generalization ability.

---

## **Outcomes**
- The model is trained and evaluated for **30 classes** of images.
- Provides accurate predictions on both the validation and test datasets.
- Capable of classifying unseen external images.
- The model is saved for future inference or deployment.

---

## **Requirements**
To run the task, ensure the following Python packages are installed:
```bash
pip install tensorflow scikit-learn matplotlib
