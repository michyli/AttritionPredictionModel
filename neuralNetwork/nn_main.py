import parse
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# Parse the data
X, y = parse.parse_csv('../EmployeeAttritionAllData.csv')

# Train:Validation:Test = 80:20:10
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.10, stratify=y, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.2222,
    stratify=y_train_val,
    random_state=42,
)

scaler = StandardScaler()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
X_train_val[numerical_cols] = scaler.fit_transform(X_train_val[numerical_cols])
X_test[numerical_cols] = scaler.fit_transform(X_test[numerical_cols])
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_val[numerical_cols] = scaler.fit_transform(X_val[numerical_cols])

# Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Initialize the MLPClassifier
mlp = MLPClassifier(
    hidden_layer_sizes=(512, 512, 512),
    activation='logistic',
    solver='adam',
    max_iter=3000,
    random_state=42,
    early_stopping=True
)

# Train the model on the resampled training data
mlp.fit(X_train_resampled, y_train_resampled)

# Predict probabilities on the validation set
y_val_pred_proba = mlp.predict_proba(X_val)[:, 1]

# Predict classes on the validation set
y_val_pred = mlp.predict(X_val)

# Evaluate the model
accuracy = accuracy_score(y_val, y_val_pred)
print("Validation Accuracy:", accuracy)

print("\nClassification Report (Validation Set):")
print(classification_report(y_val, y_val_pred))

print("Confusion Matrix (Validation Set):")
print(confusion_matrix(y_val, y_val_pred))

# ROC AUC Score
roc_auc = roc_auc_score(y_val, y_val_pred_proba)
print("ROC AUC Score (Validation Set):", roc_auc)

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_val, y_val_pred_proba)
plt.figure()
plt.plot(fpr, tpr, label='MLP Classifier (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')  # Random guessing
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Validation Set')
plt.legend(loc="lower right")
plt.show()

# Evaluate on the test set
y_test_pred_proba = mlp.predict_proba(X_test)[:, 1]
y_test_pred = mlp.predict(X_test)

test_accuracy = accuracy_score(y_test, y_test_pred)
print("\nTest Accuracy:", test_accuracy)

print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_test_pred))

print("Confusion Matrix (Test Set):")
print(confusion_matrix(y_test, y_test_pred))

# ROC AUC Score for Test Set
roc_auc_test = roc_auc_score(y_test, y_test_pred_proba)
print("ROC AUC Score (Test Set):", roc_auc_test)

# Baseline Accuracy
baseline_accuracy = y_test.value_counts(normalize=True).max()
print("\nBaseline Accuracy (Guessing the majority class):", baseline_accuracy)