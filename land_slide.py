import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Load your dataset
# Replace 'your_dataset.csv' with the path to your dataset
data = pd.read_csv("s:\\Mini project\\running code\\Dis2.csv")

# Assuming the target variable is named 'Y'
X = data.drop('Y', axis=1)
y = data['Y']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=0)
rf_classifier.fit(X_train, y_train)
rf_predictions = rf_classifier.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_precision = precision_score(y_test, rf_predictions)
rf_recall = recall_score(y_test, rf_predictions)
rf_f1 = f1_score(y_test, rf_predictions)
print("Random Forest Metrics:")
print("Accuracy:", rf_accuracy)
print("Precision:", rf_precision)
print("Recall:", rf_recall)
print("F1 Score:", rf_f1)

# XGBoost classifier
xgb_classifier = XGBClassifier(random_state=0)
xgb_classifier.fit(X_train, y_train)
xgb_predictions = xgb_classifier.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_predictions)
xgb_precision = precision_score(y_test, xgb_predictions)
xgb_recall = recall_score(y_test, xgb_predictions)
xgb_f1 = f1_score(y_test, xgb_predictions)
print("\nXGBoost Metrics:")
print("Accuracy:", xgb_accuracy)
print("Precision:", xgb_precision)
print("Recall:", xgb_recall)
print("F1 Score:", xgb_f1)

# K-Nearest Neighbors classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)
knn_predictions = knn_classifier.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_predictions)
knn_precision = precision_score(y_test, knn_predictions)
knn_recall = recall_score(y_test, knn_predictions)
knn_f1 = f1_score(y_test, knn_predictions)
print("\nKNN Metrics:")
print("Accuracy:", knn_accuracy)
print("Precision:", knn_precision)
print("Recall:", knn_recall)
print("F1 Score:", knn_f1)

# Naive Bayes classifier
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
nb_predictions = nb_classifier.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_predictions)
nb_precision = precision_score(y_test, nb_predictions)
nb_recall = recall_score(y_test, nb_predictions)
nb_f1 = f1_score(y_test, nb_predictions)
print("\nNaive Bayes Metrics:")
print("Accuracy:", nb_accuracy)
print("Precision:", nb_precision)
print("Recall:", nb_recall)
print("F1 Score:", nb_f1)

# Confusion Matrix Plot Function
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    plt.xticks([0, 1], ['Predicted 0s', 'Predicted 1s'])
    plt.yticks([0, 1], ['Actual 0s', 'Actual 1s'])
    plt.xlabel('True label')
    plt.ylabel('Predicted label')

    # Add text annotations for each cell
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.show()

# Confusion Matrices
rf_cm = confusion_matrix(y_test, rf_predictions)
xgb_cm = confusion_matrix(y_test, xgb_predictions)
knn_cm = confusion_matrix(y_test, knn_predictions)
nb_cm = confusion_matrix(y_test, nb_predictions)

# Plot Confusion Matrices
plot_confusion_matrix(rf_cm, "Random Forest Confusion Matrix")
plot_confusion_matrix(xgb_cm, "XGBoost Confusion Matrix")
plot_confusion_matrix(knn_cm, "KNN Confusion Matrix")
plot_confusion_matrix(nb_cm, "Naive Bayes Confusion Matrix")

#plot Accuracy graph
colors = ['blue', 'red', 'green', 'orange']
classifiers = ['Random Forest', 'XGBoost', 'KNN', 'Naive Bayes']
accuracies = [rf_accuracy, xgb_accuracy, knn_accuracy, nb_accuracy]
plt.figure(figsize=(8, 5))
plt.bars = plt.bar(classifiers, accuracies, color=colors)
plt.title('Accuracy Comparison')
plt.xlabel('Classifier')
plt.ylabel('Accuracy')
# Add text annotations for each bar
for i, acc in enumerate(accuracies):
    plt.text(i, acc, str(round(acc, 2)), horizontalalignment="center", verticalalignment="bottom", fontsize=10)
plt.ylim(0, 1)
plt.show()

# Plot Precision graph
colors = ['blue', 'red', 'green', 'orange']
classifiers = ['Random Forest', 'XGBoost', 'KNN', 'Naive Bayes']
precisions = [rf_precision, xgb_precision, knn_precision, nb_precision]
plt.figure(figsize=(8, 5))
plt.bars = plt.bar(classifiers, accuracies, color=colors)
plt.title('Precision Comparison')
plt.xlabel('Classifier')
plt.ylabel('Precision')
# Add text annotations for each bar in the precision graph
for i, precision in enumerate(precisions):
    plt.text(i, precision, str(round(precision, 2)), horizontalalignment="center", verticalalignment="bottom", fontsize=10)
plt.ylim(0, 1)
plt.show()

# Plot Recall graph
colors = ['blue', 'red', 'green', 'orange']
classifiers = ['Random Forest', 'XGBoost', 'KNN', 'Naive Bayes']
recalls = [rf_recall, xgb_recall, knn_recall, nb_recall]
plt.figure(figsize=(8, 5))
plt.bars = plt.bar(classifiers, accuracies, color=colors)
plt.title('Recall Comparison')
plt.xlabel('Classifier')
plt.ylabel('Recall')
# Add text annotations for each bar in the recall graph
for i, recall in enumerate(recalls):
    plt.text(i, recall, str(round(recall, 2)), horizontalalignment="center", verticalalignment="bottom", fontsize=10)
plt.ylim(0, 1)
plt.show()

# Plot F1 Score graph
colors = ['blue', 'red', 'green', 'orange']
classifiers = ['Random Forest', 'XGBoost', 'KNN', 'Naive Bayes']
f1_scores = [rf_f1, xgb_f1, knn_f1, nb_f1]
plt.figure(figsize=(8, 5))
plt.bars = plt.bar(classifiers, accuracies, color=colors)
plt.title('F1 Score Comparison')
plt.xlabel('Classifier')
plt.ylabel('F1 Score')
# Add text annotations for each bar in the F1 score graph
for i, f1_score in enumerate(f1_scores):
    plt.text(i, f1_score, str(round(f1_score, 2)), horizontalalignment="center", verticalalignment="bottom", fontsize=10)
plt.ylim(0, 1)
plt.show()

# Random Forest ROC curve
rf_probs = rf_classifier.predict_proba(X_test)[:, 1]
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
rf_auc = roc_auc_score(y_test, rf_probs)

plt.figure(figsize=(8, 6))
plt.plot(rf_fpr, rf_tpr, color='blue', lw=2, label='Random Forest (AUC = %0.2f)' % rf_auc)
plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Random Forest')
plt.legend(loc="lower right")
plt.show()

# XGBoost ROC curve
xgb_probs = xgb_classifier.predict_proba(X_test)[:, 1]
xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_probs)
xgb_auc = roc_auc_score(y_test, xgb_probs)

plt.figure(figsize=(8, 6))
plt.plot(xgb_fpr, xgb_tpr, color='red', lw=2, label='XGBoost (AUC = %0.2f)' % xgb_auc)
plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - XGBoost')
plt.legend(loc="lower right")
plt.show()

# Assuming you've already trained your Naive Bayes classifier (nb_classifier)
# Get predicted probabilities from Naive Bayes classifier
nb_probs = nb_classifier.predict_proba(X_test)[:, 1]
# Calculate the false positive rate (nb_fpr) and true positive rate (nb_tpr)
nb_fpr, nb_tpr, _ = roc_curve(y_test, nb_probs)
# Calculate AUC score for Naive Bayes
nb_auc = roc_auc_score(y_test, nb_probs)

# Plot ROC curve for Naive Bayes
plt.figure(figsize=(8, 6))
plt.plot(nb_fpr, nb_tpr, color='yellow', lw=2, label='Naive Bayes (AUC = %0.2f)' % nb_auc)
plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Naive Bayes')
plt.legend(loc="lower right")
plt.show()

# Root Mean Squared Error (RMSE)
def rmse(y_true, y_pred):
    return np.sqrt(((y_pred - y_true) ** 2).mean())

# Mean Squared Error (MSE)
def mse(y_true, y_pred):
    return ((y_pred - y_true) ** 2).mean()

# Random Forest RMSE and MSE
rf_rmse = rmse(y_test, rf_predictions)
rf_mse = mse(y_test, rf_predictions)
print("\nRandom Forest RMSE:", rf_rmse)
print("Random Forest MSE:", rf_mse)

# XGBoost RMSE and MSE
xgb_rmse = rmse(y_test, xgb_predictions)
xgb_mse = mse(y_test, xgb_predictions)
print("\nXGBoost RMSE:", xgb_rmse)
print("XGBoost MSE:", xgb_mse)

# KNN RMSE and MSE
knn_rmse = rmse(y_test, knn_predictions)
knn_mse = mse(y_test, knn_predictions)
print("\nKNN RMSE:", knn_rmse)
print("KNN MSE:", knn_mse)

# Naive Bayes RMSE and MSE
nb_rmse = rmse(y_test, nb_predictions)
nb_mse = mse(y_test, nb_predictions)
print("\nNaive Bayes RMSE:", nb_rmse)
print("Naive Bayes MSE:", nb_mse)

# Define classifiers names
classifiers = ['Random Forest', 'XGBoost', 'KNN', 'Naive Bayes']

# Define RMSE and MSE values for each classifier
rmse_values = [rf_rmse, xgb_rmse, knn_rmse, nb_rmse]
mse_values = [rf_mse, xgb_mse, knn_mse, nb_mse]

# Plotting RMSE
colors = ['blue', 'red', 'green', 'orange']
plt.figure(figsize=(10, 5))
plt.bar(classifiers, rmse_values, color=colors)
plt.xlabel('Classifier')
plt.ylabel('RMSE Value')
plt.title('Root Mean Squared Error (RMSE) for each Classifier')
plt.show()

# Plotting MSE
colors = ['blue', 'red', 'green', 'orange']
plt.figure(figsize=(10, 5))
plt.bar(classifiers, mse_values, color=colors)
plt.xlabel('Classifier')
plt.ylabel('MSE Value')
plt.title('Mean Squared Error (MSE) for each Classifier')
plt.show()

