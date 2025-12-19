# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import f1_score, ConfusionMatrixDisplay, precision_recall_fscore_support

# Generate a 3D Dataset
np.random.seed(42)
class_1 = np.random.rand(500, 3) + np.array([0.3, 0.3, 0.3])
class_2 = np.random.rand(500, 3) - np.array([0.3, 0.3, 0.3])
labels_1 = np.ones((500, 1))
labels_2 = -1 * np.ones((500, 1))
data = np.vstack((class_1, class_2))
labels = np.vstack((labels_1, labels_2))

# Save data to CSV
dataset = np.hstack((data, labels))
columns = ['Feature_1', 'Feature_2', 'Feature_3', 'Label']
df = pd.DataFrame(dataset, columns=columns)
df.to_csv('svm_dataset.csv', index=False)
print("Data table saved to 'svm_dataset.csv'.")

# Split dataset: 60/20/20
X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=0.4)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

# Initialize the SVM
svm_model = SVC(kernel='linear', C=1, max_iter=10000)

# 10-fold Cross Validation
cross_val_scores = cross_val_score(svm_model, X_train, y_train.ravel(), cv=10)

# Fit the SVM model
svm_model.fit(X_train, y_train.ravel())

# Compute F1 Score
predictions = svm_model.predict(X_test)
f1 = f1_score(y_test.ravel(), predictions)
precision, recall, _, _ = precision_recall_fscore_support(y_test.ravel(), predictions, average='binary')

# Display Confusion Matrix
ConfusionMatrixDisplay.from_estimator(svm_model, X_test, y_test.ravel())
plt.title('Confusion Matrix')
plt.show()

# Display 3D Visualization of the Points and Classification
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(class_1[:, 0], class_1[:, 1], class_1[:, 2], color='blue', label='Class +1')
ax.scatter(class_2[:, 0], class_2[:, 1], class_2[:, 2], color='red', label='Class -1')
ax.set_title('3D Data Visualization')
plt.legend()
plt.show()

# Command Window Outputs
print("Optimized Parameters: ", svm_model.coef_, svm_model.intercept_)
print(f"F1 Score: {f1}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Cross Validation Scores: {cross_val_scores}")
