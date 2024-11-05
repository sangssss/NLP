import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# Paths to the dataset (structured with folders per class in test_dir)
test_dir = 'path/to/your/test_data'

# Define models to evaluate
models_info = {
    'VGG16': {
        'model': VGG16(weights='imagenet', include_top=True),
        'preprocess': vgg_preprocess,
        'input_size': (224, 224)
    },
    'ResNet50': {
        'model': ResNet50(weights='imagenet', include_top=True),
        'preprocess': resnet_preprocess,
        'input_size': (224, 224)
    },
    'InceptionV3': {
        'model': InceptionV3(weights='imagenet', include_top=True),
        'preprocess': inception_preprocess,
        'input_size': (299, 299)
    }
}

# Prepare the test data generator
def get_test_generator(preprocess_fn, input_size):
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=input_size,
        batch_size=32,
        class_mode='categorical',  # Use 'categorical' if you have multiple classes
        shuffle=False
    )
    return test_generator

# Function to evaluate a model and calculate accuracy
def evaluate_model(model, test_generator):
    # Predict using the model
    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)  # Convert predictions to class labels
    true_classes = test_generator.classes  # Get the true labels from the generator

    # Calculate accuracy
    accuracy = accuracy_score(true_classes, predicted_classes)
    return accuracy, predicted_classes, true_classes

# Function to visualize correctly and incorrectly classified images
def visualize_results(test_generator, true_classes, predicted_classes, model_name):
    # Get a list of image file paths
    file_paths = test_generator.filepaths
    class_labels = list(test_generator.class_indices.keys())

    # Find correct and incorrect classifications
    correct_indices = np.where(predicted_classes == true_classes)[0]
    incorrect_indices = np.where(predicted_classes != true_classes)[0]

    # Plot some correctly classified images
    print(f"\nCorrectly Classified Images for {model_name}:")
    plt.figure(figsize=(12, 8))
    for i, idx in enumerate(correct_indices[:6]):  # Display first 6 correct images
        img = plt.imread(file_paths[idx])
        plt.subplot(2, 3, i + 1)
        plt.imshow(img)
        plt.title(f"True: {class_labels[true_classes[idx]]}\nPred: {class_labels[predicted_classes[idx]]}")
        plt.axis('off')
    plt.suptitle(f'Correctly Classified Images by {model_name}')
    plt.show()

    # Plot some incorrectly classified images
    print(f"\nIncorrectly Classified Images for {model_name}:")
    plt.figure(figsize=(12, 8))
    for i, idx in enumerate(incorrect_indices[:6]):  # Display first 6 incorrect images
        img = plt.imread(file_paths[idx])
        plt.subplot(2, 3, i + 1)
        plt.imshow(img)
        plt.title(f"True: {class_labels[true_classes[idx]]}\nPred: {class_labels[predicted_classes[idx]]}")
        plt.axis('off')
    plt.suptitle(f'Incorrectly Classified Images by {model_name}')
    plt.show()

# Evaluate each model, calculate accuracy, and visualize results
for model_name, info in models_info.items():
    print(f"\nEvaluating {model_name}...")
    test_generator = get_test_generator(info['preprocess'], info['input_size'])
    accuracy, predicted_classes, true_classes = evaluate_model(info['model'], test_generator)
    print(f"{model_name} Accuracy: {accuracy * 100:.2f}%")

    # Visualize the results
    visualize_results(test_generator, true_classes, predicted_classes, model_name)
