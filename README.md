# Introduction to Support Vector Machine (SVM) for Linearly Separating 3D Dataset

## Overview
This repository contains a Python implementation of a Support Vector Machine (SVM) model using `sklearn` for a beginner-level report project. **The dataset is generated to be well-structured and linearly separable to a large extent, ensuring clear classification boundaries for the SVM model.**

### What Does the Code Do?
The Python code performs the following tasks:

1. **Dataset Generation**: 
   - Creates a synthetic 3D dataset from two well-structured classes.
   - Saves the dataset to a CSV file (`svm_dataset.csv`) for easy access and reproducibility.
2. **Data Splitting**: Splits the dataset into training, validation, and testing datasets with a ratio of 60/20/20.
3. **Cross-Validation**: Evaluates the SVM model using a 10-fold cross-validation.
4. **Training with Cost Optimization**: Trains the SVM model with sub-gradient descent using cost-optimization procedures.
5. **Evaluation**:
   - Calculate F1 Score
   - Precision and Recall
   - Display Confusion Matrix
6. **Visualization**:
   - Plots the confusion matrix
   - Generates 3D visualizations of the data points and classification boundaries
7. **Command Window Outputs**: Displays optimized parameters (`w`, `b`), cross-validation scores, and performance metrics (F1 Score).

## Process Flow
### Step 1: Formulas -> Dataset
Using the formulas provided in the report:
- Margin Width (`2 / ||w||`)
- Derived sub-gradients (`w = sum(alpha_i * y_i * x_i)`), data is classified linearly.

### Step 2: Algorithm Design
The algorithm uses `sklearn.SVC` for linear classification with cost optimization (`C=1`).

### Step 3: Dataset Splitting
- **Training Set**: 60%
- **Validation Set**: 20%
- **Testing Set**: 20%

### Step 4: Cross-Validation - 10 Folds
To evaluate model stability, 10-fold cross-validation is applied.

### Step 5: Optimized Parameters
- Extract optimized hyperplane parameters (`w`, `b`).

### Step 6: Evaluation: Metrics
- Calculate F1 Score.
- Display the confusion matrix.

### Step 7: Visualization
- 3D dataset visualization
- Classification boundaries mapped to 3 axes.

## Dependencies
- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Conclusion
This linear SVM model with cost optimization achieves clear classification boundaries with robust evaluation metrics such as F1 Score. The dataset is structured to maximize linear separability.

## Generated Data
A sample of the dataset (stored as `svm_dataset.csv`) includes three features (`Feature_1`, `Feature_2`, `Feature_3`) and labels (`Label`) to distinguish the two classes (`+1`, `-1`).
