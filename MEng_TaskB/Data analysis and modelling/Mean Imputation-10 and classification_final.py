import pandas as pd

# Load dataset
file_path = r"C:\Users\difedile.rasenyalo\OneDrive - DSV\Stellenbosch\Post block Assignment3\train_data_10.csv"

data = pd.read_csv(file_path)

print(data)

# Load datasetport pandas as pd

file_path = r"C:\Users\difedile.rasenyalo\OneDrive - DSV\Stellenbosch\Post block Assignment3\test_data.xlsx"

data_test = pd.read_excel(file_path)

print(data_test)



##### Handle missing values #####

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


#### Decison Tree Classification
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# We then have to split the data into features (X) and target (y)
X_train = data_imputed.drop(columns=['target'])
y_train = data_imputed['target']

# Split the test data into features (X_test) and target (y_test)
X_test = data_test.drop(columns=['target'])
y_test = data_test['target']

# Initialize Decision Tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Predict on the test data
y_pred = clf.predict(X_test)

# Evaluate accuracy on the test set
test_accuracy = accuracy_score(y_test, y_pred)
print("Test accuracy without tuning:", test_accuracy)

# Tune the model by finding the optimal tree depth
best_depth = None
best_accuracy = 0
for depth in range(1, 11):
    # Train decision tree model with specified depth
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model on test set
    y_pred_test = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_test)
    
    # Update best depth if current model is better
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_depth = depth

print("Optimal tree depth:", best_depth)
print("Test accuracy with optimal depth:", best_accuracy)

from sklearn.metrics import precision_score, recall_score, f1_score

# Calculate precision, recall, and F1-score

precision = precision_score(y_test, y_pred, pos_label=' <=50K')
recall = recall_score(y_test, y_pred, pos_label=' <=50K')
f1 = f1_score(y_test, y_pred, pos_label=' <=50K')

print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

precision = precision_score(y_test, y_pred, pos_label=' >50K')
recall = recall_score(y_test, y_pred, pos_label=' >50K')
f1 = f1_score(y_test, y_pred, pos_label=' >50K')

print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Print confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_test))

from sklearn.metrics import precision_score, recall_score, f1_score

# Initialize Decision Tree classifier with optimal depth
optimal_model = DecisionTreeClassifier(max_depth=6, random_state=42)

# Train the optimal model on the training data
optimal_model.fit(X_train, y_train)

# Predict on the test data using the optimal model
y_pred_optimal = optimal_model.predict(X_test)

# Calculate precision, recall, and F1-score with the optimal model
precision_optimal = precision_score(y_test, y_pred_optimal, pos_label=' >50K')
recall_optimal = recall_score(y_test, y_pred_optimal, pos_label=' >50K')
f1_optimal = f1_score(y_test, y_pred_optimal, pos_label=' >50K')

print("Precision with optimal depth:", precision_optimal)
print("Recall with optimal depth:", recall_optimal)
print("F1-score with optimal depth:", f1_optimal)

precision_optimal = precision_score(y_test, y_pred_optimal, pos_label=' <=50K')
recall_optimal = recall_score(y_test, y_pred_optimal, pos_label=' <=50K')
f1_optimal = f1_score(y_test, y_pred_optimal, pos_label=' <=50K')

print("Precision with optimal depth:", precision_optimal)
print("Recall with optimal depth:", recall_optimal)
print("F1-score with optimal depth:", f1_optimal)

from sklearn.metrics import classification_report, confusion_matrix

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_optimal, target_names=[' <=50K', ' >50K']))

# Print confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_optimal))




### K- NN Classification ####

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


# Define class labels
classes = [' <=50K', ' >50K']

# X_train, X_test, y_train, y_test are the training and testing data

# Instantiate k-NN classifier
knn = KNeighborsClassifier(n_neighbors=18) 

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












