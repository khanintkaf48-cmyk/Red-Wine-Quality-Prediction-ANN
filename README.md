# Red Wine Quality Prediction Using Artificial Neural Networks

A machine-learning project that explores red wine quality data and trains an Artificial Neural Network (ANN) to classify wine quality from physicochemical measurements.

## Project overview

The notebook follows an end-to-end workflow:

1. Load and inspect the red wine quality dataset.
2. Explore feature distributions, missing values, outliers, and target-class balance.
3. Use a Random Forest model to examine feature importance.
4. Remove two features (`free sulfur dioxide` and `citric acid`) during feature selection.
5. Split the data into training and testing sets and standardize the input features.
6. Apply SMOTE to the training set to address class imbalance.
7. Encode the target labels and train a multi-layer ANN classifier.
8. Evaluate predictions using accuracy and a classification report.

## Dataset

- **File:** `winequality-red.csv`
- **Dataset:** Red Wine Quality
- **Rows:** 1,599
- **Original input features:** 11 physicochemical measurements
- **Target:** `quality` (the wine quality score)

The dataset is included in this repository. The notebook uses the CSV file from the project directory.

## Model

The notebook builds a Sequential ANN with:
- An input layer for 9 selected features
- Dense hidden layers with ReLU activation
- Dropout layers to help reduce overfitting
- A 6-unit Softmax output layer for the six quality classes represented in the training workflow
- Adam optimizer and categorical cross-entropy loss

SMOTE is applied only to the training data before model training.

## Result

The saved notebook reports a test accuracy of **69.375%** for the trained ANN on its test split. Results may vary if the notebook, library versions, or training setup are changed.

## Tech stack

- Python
- Jupyter Notebook
- NumPy and Pandas
- Matplotlib and Seaborn
- Scikit-learn
- Imbalanced-learn (SMOTE)
- TensorFlow / Keras

## Project structure

```text
Red-Wine-Quality-Prediction-ANN/
├── winequality-red.csv
├── wine_quality_pred_ANN.ipynb
├── 434px-Relationship_between_mean_and_median_under_different_skewness.png
├── LICENSE
└── README.md
```

## Getting started

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Red-Wine-Quality-Prediction-ANN
```

Replace `<your-repository-url>` with the URL of your GitHub repository.

### 2. Install the dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn tensorflow jupyter
```

### 3. Start Jupyter Notebook

```bash
jupyter notebook
```

Open `wine_quality_pred_ANN.ipynb` and run the notebook cells in order.

## Notes

- The notebook is the main project implementation.
- The dataset and supporting image are included in the repository.
- This repository documents an educational machine-learning project; it is not presented as a production wine-testing system.

## License

See the `LICENSE` file included in this repository.
