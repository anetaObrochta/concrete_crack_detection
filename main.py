from src.data import preprocess
from src.models import train_model, predict_model
from src.visualization import visualize

def main():
    # Load and preprocess data
    data = preprocess.load_data()
    preprocessed_data = preprocess.preprocess_images(data)

    # Train model
    model = train_model.train(preprocessed_data)

    # Make predictions
    predictions = predict_model.predict(model, new_data)

    # Visualize results
    visualize.plot_results(predictions)

if __name__ == "__main__":
    main()