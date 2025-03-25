import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay 
from lazypredict.Supervised import LazyClassifier
import matplotlib.pyplot as plt
import joblib

df = pd.read_csv('preprocessedData.csv')

print(df.head())

# Split data into features and target
x_data = df.drop('cluster', axis=1)
y_data = df['cluster']

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=11)

# Initialize LazyClassifier
classifier = LazyClassifier(verbose=0, ignore_warnings=False, custom_metric=None)

# Fit the classifier
models, predictions = classifier.fit(x_train, x_test, y_train, y_test)

# Print model performance metrics
print(models)

### CREATE A CLASSIFIER THAT PREDICTS THE CLUSTER OF A STOCK using NearestCentroid as it is the best performing model in LazyClassifier ###

# Initialize Nearest Centroid Classifier
nc_classifier = NearestCentroid(metric="euclidean")

# Train the classifier
nc_classifier.fit(x_train, y_train)

# Make predictions on the test set
y_pred = nc_classifier.predict(x_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print classification report
print(classification_report(y_test, y_pred))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

'''
# Save the classifier
joblib.dump(nc_classifier, 'nc_classifier.joblib')
'''