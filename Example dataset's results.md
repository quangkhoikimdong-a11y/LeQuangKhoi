# Results and Analysis: Support Vector Machine (SVM) for Linearly Separable 3D Dataset

## Confusion Matrix
The confusion matrix displays the performance of our SVM model in predicting the labels for the testing dataset.

![Confusion Matrix](main/confusion_matrix.png)

### Interpretation:
- **True Positives (97)**: Instances correctly classified as `+1`.
- **True Negatives (97)**: Instances correctly classified as `-1`.
- **False Positives (2)**: `-1` instances misclassified as `+1`.
- **False Negatives (4)**: `+1` instances misclassified as `-1`.

The low number of false positives and false negatives indicates that the model performs well in distinguishing between the two classes.

---

## 3D Data Visualization
The 3D scatter plot illustrates the dataset points and their classification.

![3D Data Visualization](main/data_visualization.png)

### Features:
- The dataset is linearly separable to a large extent, as seen in the classification visualization.
- The points belonging to `Class +1` are assigned the color blue, and those belonging to `Class -1` are red.

---

## Performance Metrics
The following metrics provide detailed insights into the performance of the model:

- **Optimized Parameters (w, b)**: 
  - `w` = [[3.42635865, 2.27472136, 3.05506417]]
  - `b` = [-4.4027234]
  
  These parameters are the coefficients of the hyperplane that optimally separates the two classes.

- **F1 Score**: 0.97  
  A high F1 score reflects the harmonic mean of precision and recall, ensuring a balanced performance across positive and negative predictions.

- **Precision**: 0.9797979797979798  
  The fraction of positive predictions that are correct.

- **Recall**: 0.9603960396039604  
  The fraction of positive instances correctly identified.

- **Cross-Validation Average Score**: 0.95  
  Stability of the model is demonstrated by consistently high scores in 10-fold cross-validation.

---

## Satisfactory Model Performance
Based on the metrics:
- The confusion matrix demonstrates a high level of accuracy in classification.
- Precision and Recall values are close to 1.0, indicating high reliability in detecting both classes with minimal error.
- The visualization of the 3D dataset shows clear separation between `Class +1` and `Class -1`.

Overall, the model performs satisfactorily, with excellent predictive capabilities and stability during cross-validation.
