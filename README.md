<p align="center">
      <img src="https://i.ibb.co/nCd1n4k/logmeal-logo-2.png">
</p>

<p align="center">
   <img src="https://img.shields.io/badge/Pandas-lavender" alt="Pandas">
   <img src="https://img.shields.io/badge/NumPy-thistle" alt="NumPy">
   <img src="https://img.shields.io/badge/Matplotlib-lightcyan" alt="Matplotlib">
   <img src="https://img.shields.io/badge/Torch-thistle" alt="Torch">
   <img src="https://img.shields.io/badge/CNN-lightcyan" alt="CNN">
   <img src="https://img.shields.io/badge/Torchvision-lavender" alt="Torchvision">
</p>

## About

Detection of inconsistencies in food descriptions in online food ordering and delivery platform serves as an important ingredient for success, customer retention and satisfaction. Most companies providing online food ordering and delivery platforms are gradually utilising deep learning based solutions to detect if a food image shown on their platform conforms to the description given or category.

## Documentation

### Introduction
The project "Food image recognition" aims to enhance the quality of services in the online food ordering and delivery platform. One crucial element for success in this domain is ensuring the correspondence between the displayed image of a dish, its description, and category. This project employs deep neural networks for automatic detection of inconsistencies.

### Overview


### Goals and Objectives
  * Development of a Deep Learning Model: The project proposes creating a Convolutional Neural Network (CNN) model trained on the "Food101" dataset for food image classification.

  * Performance Enhancement of the Model: During the training of the base CNN model (provided in Table 1), three techniques studied in the deep learning course are applied to improve performance. Examples of such techniques include dropout, batch normalization, and others to boost the model's efficacy.

  * Performance Monitoring using Tensorboard: To track the training process and evaluate the model, Tensorboard is utilized. Metrics such as loss, accuracy, and F1 score are logged during both training and validation.

  * Application of Transfer Learning for Improved Performance: In the final part of the project, transfer learning is applied using the pre-trained DENSENET121 model to achieve higher performance compared to the enhanced base CNN model.


### Technologies and Tools
  * Python
  * PyTorch
  * Tensorboard
  * scikit-learn
  * tqdm
  * torchvision
  * numpy

### Baseline Model (FoodCNN)
The baseline model is a convolutional neural network (CNN) designed for food image classification. The architecture includes convolutional layers, max-pooling layers, and fully connected layers. The model is trained on the Food101 dataset using techniques such as dropout, batch normalization, and more to enhance performance.

### Model Architecture
The architecture of the baseline model is as follows:
| Layer | Layer Type | Kernel size | Stride | Padding | Out channels |
|-------|------------|-------------|--------|---------|--------------|
| 0     | Input      | -           | -      | -       | 3            |
| 1     | Convolutional | 3 x 3     | 1      | 1 x 1   | 10           |
| 2     | Convolutional | 3 x 3     | 1      | 1 x 1   | 10           |
| 3     | Max-pooling | 2 x 2       | 2      | -       | -            |
| 5     | Convolutional | 3 x 3     | 1      | 1 x 1   | 10           |
| 6     | Convolutional | 3 x 3     | 1      | 1 x 1   | 10           |
| 7     | Max-pooling | 2 x 2       | 2      | -       | -            |
| 8     | Flatten    | -           | -      | -       | -            |
| 9     | Fully connected | -       | -      | -       | 2560 (output)|
| 10    | Fully connected (softmax) | - | - | -       | 101 (output) |

### Training
The baseline model is trained on 10 epochs using the SGD optimizer with a learning rate of 0.001. The training progress is logged to TensorBoard, and metrics such as training and validation loss, accuracy, and F1 score are monitored.

### Improved Model (FoodCNN2)
The improved model (FoodCNN2) incorporates additional techniques, including batch normalization and dropout layers, to enhance training stability and prevent overfitting. This model aims to achieve better performance than the baseline.

### Model Architecture
The architecture of the improved model is based on the baseline model with added batch normalization and dropout layers.

### Training
Similar to the baseline, the improved model is trained on 10 epochs using the SGD optimizer. The training progress is logged to TensorBoard, and metrics are monitored.

### Further Enhanced Model (FoodCNN3)
The further enhanced model (FoodCNN3) introduces deeper layers, global average pooling, and increased units in fully connected layers for improved feature extraction and discrimination.

### Model Architecture
The architecture of the further enhanced model includes deeper convolutional layers, global average pooling, and modifications to fully connected layers.

### Training
The further enhanced model is trained on 10 epochs, and its training progress is logged to TensorBoard. Metrics such as training and validation loss, accuracy, and F1 score are monitored.

### Transfer Learning Model (DENSENET121)
The transfer learning model utilizes the DENSENET121 architecture, a pre-trained model on a large dataset, for improved performance on the food image classification task.

### Model Architecture
The transfer learning model modifies the fully connected layer of DENSENET121 to match the number of classes in the Food101 dataset.

### Training
The model is trained on a subset of the Food101 dataset, and its performance is evaluated on the validation set. Training progress is logged to TensorBoard, including metrics like training and validation loss, accuracy, and F1 score.

### Conclusion
  * The baseline models (FoodCNN, FoodCNN2, FoodCNN3) provided a starting point for food image classification, with varying levels of complexity and performance.

  * FoodCNN2, with added batch normalization and dropout, showed improvements over the basic FoodCNN model.

  * FoodCNN3, incorporating deeper layers and global average pooling, aimed for enhanced feature extraction and discrimination.

  * The transfer learning model based on DENSENET121 demonstrated the potential of leveraging pre-trained models for improved classification performance on a smaller dataset.

  * The choice of model architecture, regularization techniques, and transfer learning significantly impacted the final classification results.

  * Further experimentation, hyperparameter tuning, and exploration of other pre-trained models could lead to even better performance in food image classification tasks.

  * The documented code and training process provide a foundation for future work and experimentation in the domain of food image classification using deep learning.

## Developers

- Kamyshnikov Dmitrii :
      - [GitHub](https://github.com/kama34)
      - [Email](mailto:kamyshnikovdmitri@yandex.ru)
      - [Telegram](https://t.me/+79101663108)

## License
Project kama34.FoodImageRecognition is distributed under the MIT license.
