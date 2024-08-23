


# Your First AI Application: Flower Species Classifier

This project is a deep learning-based image classifier designed to recognize different species of flowers. It uses a pre-trained MobileNetV2 model, fine-tuned on the Oxford Flowers 102 dataset, to predict the species of a flower given its image. The classifier can be integrated into various applications, such as a smartphone app that identifies flowers in real-time.

## Project Overview

Artificial Intelligence (AI) is becoming increasingly integrated into everyday applications. One common use case is image classification, where a deep learning model is trained on a large dataset of images to recognize and categorize them. In this project, I've built an AI application that can classify images of flowers into one of 102 species.

This project was provided as the final project in the **Udacity Intro to TensorFlow section of the Intro to AI Programming with TensorFlow Nanodegree**. The entire application is developed and executed within a Jupyter Notebook, offering an interactive environment to experiment with the code and visualize results.

The project is broken down into several key steps:
1. **Load and Preprocess the Dataset:** Load the Oxford Flowers 102 dataset and prepare it for training by resizing and normalizing the images.
2. **Build and Train the Classifier:** Utilize the MobileNetV2 pre-trained model from TensorFlow Hub and add a custom feed-forward network on top for classification.
3. **Model Evaluation:** Evaluate the model’s performance on unseen test data to ensure it generalizes well.
4. **Inference:** Implement a function to predict the species of a flower given a new image and display the results.
5. **Visualization:** Visualize the model's predictions alongside the input images.

## Dataset

The Oxford Flowers 102 dataset consists of images of 102 different types of flowers. The dataset is divided into three splits: training, validation, and testing. Images are resized to 224x224 pixels and normalized to meet the input requirements of the MobileNetV2 model.

## Key Features

- **Pre-trained Model:** The project uses the MobileNetV2 architecture, which is lightweight and optimized for mobile devices.
- **Custom Classifier:** A custom dense network is added on top of the pre-trained model to classify the flower images into 102 categories.
- **Data Augmentation:** The dataset is preprocessed to include normalization and resizing, which helps improve model accuracy.
- **Visualization:** The results of the model are visualized using Matplotlib, showing both the input image and the predicted probabilities.

## Installation

To run this project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/flower-classifier.git
   cd flower-classifier
   ```

2. **Install the necessary packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset:**
   The dataset is automatically loaded using TensorFlow Datasets, so no manual download is required.

4. **Run the Jupyter Notebook:**
   ```bash
   jupyter notebook Project_Image_Classifier_Project..ipynb
   ```

5. **Make predictions:**
   Execute the cells in the Jupyter Notebook to train the model and make predictions.

## Usage

This project can be extended to classify images in various contexts, not just flowers. The model can be retrained on any labeled image dataset and deployed in different environments, such as mobile apps, web applications, or embedded systems.

### Example Usage

Here’s how to use the classifier to predict the species of a flower:

```python
from predict import predict

image_path = './test_images/orange_dahlia.jpg'
model_path = './flower_classifier.hdf5'

# Predict the top 5 most likely classes
probs, classes = predict(image_path, model_path, top_k=5)
print(probs)
print(classes)
```

## Results

The model achieves a validation accuracy of over 75% after training. On the test set, the accuracy is approximately 71%, which demonstrates the model's capability to generalize to unseen images.

## Visualizations

The project includes visualization functions that display the input image alongside the model’s top 5 predictions, including the probabilities for each class. This helps to better understand the model's performance.

## Future Work

- **Deployment:** Integrate the model into a mobile or web application.
- **Model Optimization:** Experiment with other pre-trained models or fine-tuning strategies to improve accuracy.
- **Additional Datasets:** Extend the model to work with other datasets, such as animals or everyday objects.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

