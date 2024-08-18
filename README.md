
# CONCRETE CRACK DETECTOR ML MODEL

This project implements a machine learning model designed to detect cracks in concrete surfaces from images. Utilizing a Convolutional Neural Network (CNN) trained on a dataset of 10,000 concrete surface images, the model can accurately classify images as either containing cracks or not.

Key features:
- Binary classification: crack detected or no crack detected - Achieved 96% Accuracy during training, validation, and testing
- Input: 227x227 pixel RGB images of concrete surfaces
- Output: Prediction with confidence score
- Web-based interface for easy testing and visualization
- Includes model training statistics and performance metrics


To view, test, and visualize the training of the Concrete Crack Detector model:

https://huggingface.co/spaces/aobrochta/concrete_crack_detector

This tool aims to assist in infrastructure maintenance by automating the initial screening of concrete surfaces for potential damage, improving efficiency and consistency in structural inspections.

## Example use of this model:
- ** Bridge Inspection **
Analyze images from routine bridge inspections to quickly identify areas needing closer examination.
Integrate with drone-based inspection systems for automated crack detection in hard-to-reach areas.


- ** Building Safety Assessment **
Assess the structural integrity of buildings, particularly in earthquake-prone regions.
Conduct rapid post-earthquake damage assessments to prioritize response efforts.


- ** Road and Highway Maintenance **
Monitor the condition of concrete highways, identifying early signs of wear and potential hazards.
Assist in planning preventive maintenance schedules based on detected crack patterns.




## Input Images for Testing
You can use the provided images below the image uploader to test the model.
Alternatively, this model works with images from other online sources, but they need to be
cropped to only the concrete surface area. The model is not trained on images that have other artifacts such as grass, sidewalk outlines, or objects.

## Application Overview

1. There are 2 tabs on top – “Crack Detection” and “Training Statistics”.
2.	Below the image upload, there are positive and negative images that can be used to test the model. 
3. Download an image by clicking the download icon in the top right corner of the image.
4. Once downloaded, drag it from your downloads and drop it into the image uploaded. Instead of drag and drop, you can also click upload in the center of the uploaded image and select the downloaded image.
5. Alternatively, you can test with more images from the additional_test_input folder in the project directory in the ‘data’ subdirectory. 
6. Once the image is uploaded, Click the Predict button below the image uploader. The model predictor will start the processing on the right side. You will see a time elapsed in seconds, showing it is being processed. Once it finishes, it will display its prediction.
7. The result will be written in the Prediction box - either ‘Crack detected’ or ‘No crack detected.’ Next to it will be its confidence score. The closer it is to 1, the more confident it is that there’s a crack in the image. The closer to 0, it’s more confident that there is no crack present.
8. Below the prediction box is the confidence meter showing the model’s prediction confidence.
9. To use a different image, click the x button in the top right corner of the uploaded image and reload a different image. Click the Predict button to see the new prediction.
10. To explore visualizations of the training process, click on the second tab, ‘Training Statistics’, at the top of the page, on the right side of the ‘Crack Detection’ tab.

## Data Processing and Model Training
Data for training was obtained from https://www.kaggle.com/datasets/arnavr10880/concrete-crack-images-for-classification

The saved model can be found in the project directory in the 'models' subdirectory.

To view my Jupyter Notebooks with saved outputs, they are found in the 'notebook' subdirectory in main project directory.

### To recreate the same data preprocessing used to train the concrete crack detection model
- First download the raw data from the link above.


- Use the environment:

  - `conda env create -f environment.yml`
  - `conda activate concrete_crack_detection`


- Save the file as 'raw' and place it in the project directory at subdirectory 'data'.


- To create the smaller sample dataset that will be used for training, run this command in IDE terminal:
`python create_sample_dataset.py`


- To create the processed dataset that has the training, validation, and testing split.
This will create a processed folder in the data subdirectory with the NuMpy Arrays.
Run this command in the IDE terminal:
`python data_loading_splitting.py`


- To extract additional images to be later used on the saved model and for further testing.
This will take 50 positive and 50 negative images from the raw dataset, excluding the ones that are in sample dataset.
run this command in the IDE terminal:
`python extract_additional_images.py`


- To open Jupyter Notebooks in Anaconda Prompt, run command:
`Jupyter Notebook`

  - This will open a Home directory of the project.
Navigate to the Notebooks folder. This folder will have:
  1. `01_data_exploration.ipynb` - for data exploration and verification prior to training
  2. `concrete_crack_detection_v4.ipynb` - the main model notebook for data loading, model training, testing
  3. `user_test_model.ipynb` - for user testing with ability to upload an image for new predictions by the saved model


