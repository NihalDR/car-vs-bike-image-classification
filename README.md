# car-vs-bike-image-classification

This project is a **binary image classification model** built using **Convolutional Neural Networks (CNNs)** in TensorFlow/Keras. It classifies images into two categories: **Car** or **Bike**. The model is trained on a custom dataset and achieves high accuracy using a simple yet powerful CNN architecture.

---

## 📂 Dataset Info

* ✅ **Dataset is included in this repository** under the folder: `archive/Car-Bike-Dataset`

* ❌ **Model file is not included** to keep the repo size light. You must **run the training script once** to generate and save the model (`.h5` format).

* Structure:

  ```
  archive/Car-Bike-Dataset/
  ├── Car/
  └── Bike/
  ```

* Images are automatically split into **80% training** and **20% validation** using TensorFlow's `image_dataset_from_directory`.

---

## 🧠 Model Architecture

The model is a Sequential CNN with the following layers:

* 3 Convolutional layers with ReLU activation
* MaxPooling after each convolution
* Flatten layer
* Dense layer with 128 units and dropout
* Final dense layer with **1 neuron (sigmoid)** for binary classification

```python
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256,256,3)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])
```

---

## ⚙️ Training Details

* **Image Size:** 256x256
* **Batch Size:** 32
* **Loss Function:** `binary_crossentropy`
* **Optimizer:** `adam`
* **Epochs:** 10
* **Normalization:** Rescaling pixel values to `[0, 1]`

---

## 🖼️ Prediction on New Image

Load and preprocess an image:

```python
img = cv2.imread('test2.jpg')
resize = tf.image.resize(img, (256, 256))
yhat = model.predict(np.expand_dims(resize / 255, 0))
```

Prediction logic:

```python
if yhat < 0.5:
    print("Predicted class is a bike")
else:
    print("Predicted class is a car")
```

---

## 💾 Model Saving & Loading

```python
model.save('models/binaryimageclassifiernewversionlive.h5')
new_model = load_model('models/binaryimageclassifiernewversionlive.h5')
```

> ⚠️ The model (`.h5` file) is **not pre-included** in this repo. You need to **train the model** and save it locally.

---

## ✅ Requirements

* Python 3.8+
* TensorFlow 2.x
* NumPy
* OpenCV
* Matplotlib

Install all dependencies:

```bash
pip install tensorflow opencv-python matplotlib numpy
```

---

## 🚀 Future Improvements

* Add a simple frontend using Streamlit or Flask
* Integrate model evaluation metrics (confusion matrix, precision, recall)
* Export model as a `.tflite` file for mobile deployment

---

## 🙌 Acknowledgements

* TensorFlow & Keras
* Car vs Bike dataset (included in repo)

---

## 📌 License

This project is licensed under the [MIT License](LICENSE).
