# ResNet Image Classifier

This project implements a Residual Neural Network (ResNet) model to classify images from the CIFAR-10 dataset. The implementation showcases the effectiveness of ResNets in handling the vanishing gradients problem in deep neural networks, thus allowing the training of very deep networks.

### Logic of Residual Connections

<img src="https://github.com/ojasdeshpande10/ResNet/assets/51486220/c48c15fe-cd75-4a43-be1b-53e88ea034e5" width="250">


### The complete architecture of the 34 layer Residual Network as presented in this paper [https://arxiv.org/pdf/1512.03385]
<img src="https://github.com/ojasdeshpande10/ResNet/assets/51486220/ef252474-70d1-495d-8cd4-55c2a3ab1fd9" width="250">


## Project Structure

Here's an overview of the main components of our project and what each file is responsible for:

- **main.py**
  - The entry point of the program. This script runs the whole process, from data loading and processing to training the model and evaluating results.

- **DataReader.py**
  - Contains functionality for loading and preprocessing data. It ensures that data is formatted and ready for analysis or input into the model.

- **Model.py**
  - Defines the machine learning or statistical model. This includes the architecture of the model, its parameters, and any other relevant settings.

- **ImageUtils.py**
  - Provides utilities for image processing tasks such as image resizing, normalization, augmentation, etc., which are crucial for image data preparation.

- **Network.py**
  - Manages the neural network operations, including training loops, loss computation, and backpropagation. It is responsible for setting up the network and updating weights.

## Getting Started

Instructions on how to setup and run the project, including installing necessary packages, setting up the environment, etc.

```bash
# Example command to run the project
python main.py



