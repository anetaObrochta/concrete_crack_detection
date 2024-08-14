import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import gradio as gr

# Loads the model
model = tf.keras.models.load_model('models/concrete_crack_detection_v4.h5')


def preprocess_image(image):
    image = image.resize((227, 227))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)


def create_confidence_meter(confidence):
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.bar(['No Crack', 'Crack'], [1 - confidence, confidence], color=['blue', 'red'])
    ax.set_ylim(0, 1)
    ax.set_title('Prediction Confidence')
    ax.set_ylabel('Confidence Score')
    plt.close(fig)
    return fig


def predict_crack(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)

    confidence = float(prediction[0][0])
    result = "Crack detected" if confidence > 0.5 else "No crack detected"

    confidence_meter = create_confidence_meter(confidence)

    return f"{result} (Confidence: {confidence:.2f})", confidence_meter


# Custom CSS to maintain consistent size
# Minimal custom CSS
custom_css = """
.container {max-width: 800px; margin: auto;}
"""


def load_test_images():
    test_images = []
    for class_name in ['positive', 'negative']:
        path = f'data/additional_testing_input/{class_name}'
        files = os.listdir(path)
        sample_files = np.random.choice(files, 6, replace=False)
        for file in sample_files:
            img_path = os.path.join(path, file)
            test_images.append((img_path, class_name))
    return test_images


# Create the main prediction interface combined with test images
with gr.Blocks(css=custom_css) as prediction_interface:
    with gr.Column(elem_classes="container"):
        gr.Markdown("# Concrete Crack Detector")
        gr.Markdown("Upload an image of concrete to detect if it contains a crack.")

        with gr.Row():
            with gr.Column(scale=2):
                input_image = gr.Image(type="pil", label="Upload Image", height=300)
                predict_btn = gr.Button("Predict")

            with gr.Column(scale=1):
                output_text = gr.Textbox(label="Prediction")
                output_plot = gr.Plot(label="Confidence Meter")

        gr.Markdown("## Test Images")
        gr.Markdown(
            "These images are from the raw dataset and were not used in training. You can download and use them to test the model.")

        test_image_list = load_test_images()
        with gr.Row():
            for img_path, class_name in test_image_list:
                gr.Image(img_path, label=f"{class_name.capitalize()} Sample", height=150)

        predict_btn.click(
            fn=predict_crack,
            inputs=input_image,
            outputs=[output_text, output_plot]
        )

# Load paths to static visualization images
accuracy_plot = "Visualizations/Model_Accuracy.png"
loss_plot = "Visualizations/Model_Loss.png"
confusion_matrix = "Visualizations/Confusion_Matrix.png"
classification_report = "Visualizations/Classification_Report.png"
data_distribution = "Visualizations/Distribution_Bar_Graph_Train_Val_Test.png"
data_exploration_image = "Visualizations/Sample_Images_Training_Set.png"

# Create the training statistics interface
with gr.Blocks() as training_stats:
    gr.Markdown("## Model Training Statistics")
    with gr.Row():
        gr.Image(accuracy_plot, label="Model Accuracy")
        gr.Image(loss_plot, label="Model Loss")
    with gr.Row():
        gr.Image(confusion_matrix, label="Confusion Matrix")
        gr.Image(data_distribution, label="Data Distribution")
    with gr.Row():
        gr.Image(classification_report, label="Classification Report")

    gr.Markdown("## Sample Image from Data Exploration")
    gr.Image(data_exploration_image)

# Create the test images interface
with gr.Blocks() as test_images:
    gr.Markdown("## Additional Test Images")
    gr.Markdown(
        "These images are from the raw dataset and were not used in training. You can download and use them to test "
        "the model.")

    test_image_list = load_test_images()
    with gr.Row():
        for img_path, class_name in test_image_list:
            gr.Image(img_path, label=f"{class_name.capitalize()} Sample")

# Combine interfaces
demo = gr.TabbedInterface(
    [prediction_interface, training_stats],
    ["Crack Detection", "Training Statistics"]
)

if __name__ == "__main__":
    demo.launch()

