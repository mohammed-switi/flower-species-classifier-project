import argparse
import json
import numpy as np
import tensorflow as tf
from PIL import Image
import tensorflow_hub as hub

class Rescaling(tf.keras.layers.Layer):
    def __init__(self, scale):
        super(Rescaling, self).__init__()
        self.scale = scale

    def call(self, inputs):
        return inputs * self.scale

    def get_config(self):
        config = super(Rescaling, self).get_config()
        config.update({'scale': self.scale})
        return config

def load_model(model_path):
    custom_objects = {
    'KerasLayer': hub.KerasLayer
}

    loaded_model = tf.keras.models.load_model('flower_classifier.hdf5', custom_objects=custom_objects)


    return loaded_model


def process_image(image):
    # Resize image to 224x224
    image = tf.image.resize(image, (128, 128))
    # Normalize image
    image = image / 255.0
    return image

def predict(image_path, model, top_k=5):
    image = Image.open(image_path)
    processed_image = process_image(np.asarray(image))
    processed_image = np.expand_dims(processed_image, axis=0)
    
    predictions = model.predict(processed_image)
    top_k_indices = np.argsort(predictions[0])[-top_k:][::-1]
    
    top_k_probs = [predictions[0][i] for i in top_k_indices]
    top_k_classes = [str(i) for i in top_k_indices]
    
    return top_k_probs, top_k_classes

def main():
    parser = argparse.ArgumentParser(description="Predict flower class from an image.")
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('model_path', type=str, help='Path to the saved Keras model')
    parser.add_argument('--top_k', type=int, default=5, help='Return the top K most likely classes')
    parser.add_argument('--category_names', type=str, help='Path to a JSON file mapping labels to flower names')
    
    args = parser.parse_args()
    
    model = load_model(args.model_path)
    
    probs, classes = predict(args.image_path, model, args.top_k)
    
    if args.category_names:
        with open(args.category_names, 'r') as f:
            class_names = json.load(f)
        class_names = [class_names.get(str(cls), "Unknown") for cls in classes]
    else:
        class_names = classes
    
    print("Top K Predictions:")
    for i in range(len(probs)):
        print(f"Class: {class_names[i]}, Probability: {probs[i]:.4f}")

if __name__ == '__main__':
    main()
