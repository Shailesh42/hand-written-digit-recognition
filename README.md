# Handwritten Digit Recognition using CNN in MATLAB

This project implements a Convolutional Neural Network (CNN) for handwritten digit recognition using the MNIST dataset. The project includes data preprocessing, model training, validation, and testing.

---

## **Features**
- **Preprocessing:**
  - Normalize pixel values (0 to 255 scaled to 0 to 1).
  - Reshape images to 28x28 for CNN compatibility.
  - Split data into training and validation sets.
- **CNN Model:**
  - Includes convolutional, ReLU, max-pooling, dropout, and fully connected layers.
- **Evaluation Metrics:**
  - Validation accuracy, confusion matrix, precision, recall, and F1-score.
- **Test Data:**
  - Predictions for test data saved as images.

---

## **Project Structure**
- `handwritten_digit_recognition.m`: Main MATLAB script to train and test the CNN.
- `train.csv`: Contains training data with labels and pixel values.
- `test.csv`: Contains test data with pixel values.
- `digitRecognizerModel.mat`: Trained CNN model (optional).
- `TestImages/`: Folder with test data images saved as `.png` files.
- `README.md`: Documentation for the project.

---

## **Setup Instructions**
### **Requirements**
- MATLAB with Deep Learning Toolbox.
- MNIST dataset (`train.csv` and `test.csv`).

### **Steps**
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/handwritten-digit-recognition.git
   cd handwritten-digit-recognition
