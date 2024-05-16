import pandas as pd


# Load dataset
file_path = r"C:\Users\difedile.rasenyalo\OneDrive - DSV\Stellenbosch\Post block Assignment3\train_data_10.csv"

data = pd.read_csv(file_path)

print(data)

# Load dataset
file_path = r"C:\Users\difedile.rasenyalo\OneDrive - DSV\Stellenbosch\Post block Assignment3\test_data.xlsx"

data_test = pd.read_excel(file_path)

print(data_test)



# Handle missing values
data.fillna({'fnlwgt': data['fnlwgt'].mean(), 'capital-loss': 0, 'hours-per-week': data['hours-per-week'].mean()}, inplace=True)
data['target'].fillna(data['target'].mode()[0], inplace=True)

# Mean imputation function
def mean_imputation(column):
    if pd.api.types.is_numeric_dtype(column):  # Check if column is numeric
        mean = column.mean()  # Calculate the mean value of the non-missing values
        return column.fillna(mean)  # Replace missing values with the mean
    else:
        mode_value = column.mode()[0]  # Calculate the mode value of the categorical column
        return column.fillna(mode_value)  # Replace missing values with the mode value

# Apply mean imputation to each column
data_imputed = data.apply(mean_imputation, axis=0)

# Export DataFrame with mean imputed values to Excel
data_imputed.to_csv("mean_imputed_data_10.csv", index=False)

print("DataFrame with mean imputed values exported to mean_imputed_data_10.csv")



from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# data_imputed contains the mean-imputed DataFrame. This data frame will be used to train the model 

# We then have to split the data into features (X) and target (y)
X_train = data_imputed.drop(columns=['target'])
y_train = data_imputed['target']

# Split the test data into features (X_test) and target (y_test)
X_test = data_test.drop(columns=['target'])
y_test = data_test['target']

# Initialize Decision Tree classifier
clf = DecisionTreeClassifier(random_state=42) #The value 42 is an arbitrary choice; you can use any integer value or leave it unspecified if you don't need reproducibility.

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Predict on the test data
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Print confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

#y_test and y_pred contain the true labels and predicted labels respectively
# Define class labels
classes = sorted(set(y_test) | set(y_pred))

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=classes)


# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=classes, yticklabels=classes, cbar=False, annot_kws={"color": 'black'})
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


#### K-NN classification model 
from sklearn.neighbors import KNeighborsClassifier
# Instantiate k-NN classifier
knn = KNeighborsClassifier(n_neighbors=18)  # You can adjust n_neighbors as needed

# Train the classifier
knn.fit(X_train, y_train)

# Make predictions
y_pred_knn = knn.predict(X_test)

# Evaluate the model
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("Accuracy:", accuracy_knn)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_knn))

# Plot confusion matrix
cm_knn = confusion_matrix(y_test, y_pred_knn)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_knn, annot=True, cmap='Blues', fmt='d', xticklabels=classes, yticklabels=classes, cbar=False, annot_kws={"color": 'black'})
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - k-NN Classifier')
plt.show()


### K- NN Classification 
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# Define a range of k values
k_values = range(1, 20)

# Initialize empty list to store cross-validation scores
cv_scores = []

# Perform cross-validation for each value of k
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')  # 5-fold cross-validation
    cv_scores.append(scores.mean())

# Find the optimal value of k with the highest cross-validation score
optimal_k = k_values[cv_scores.index(max(cv_scores))]

print("Optimal k:", optimal_k)

###  K- NN Classification 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Assuming X_train, X_test, y_train, y_test are the training and testing data

# Instantiate k-NN classifier
knn = KNeighborsClassifier(n_neighbors=16)  # You can adjust n_neighbors as needed

# Train the classifier
knn.fit(X_train, y_train)
knn.fit(X_test, y_test)

y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test, y_pred)


# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Greens', fmt='d', xticklabels=classes, yticklabels=classes, cbar=False, annot_kws={"color": 'black'})
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - k-NN Classifier')
plt.show()

### Evaluate the model 

accuracy_knn = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy_knn)
print(classification_report(y_test, y_pred))





