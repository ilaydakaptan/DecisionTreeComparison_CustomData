# USING THE DECISION TREE ALGORITHM FROM LIBRARY

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# Load the dataset
df = pd.read_excel('modified_glassesData.xlsx')

# Separate features and target variable
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Initialize StratifiedKFold for 10-fold cross-validation
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Create an empty list to store validation scores
validation_scores = []

# Perform 10-fold cross-validation
for i, (train_index, test_index) in enumerate(kf.split(X, y), 1):
    X_train, X_val = X.iloc[train_index], X.iloc[test_index]
    y_train, y_val = y.iloc[train_index], y.iloc[test_index]

    # Build the decision tree using scikit-learn
    decision_tree = DecisionTreeClassifier(random_state=42)
    decision_tree.fit(X_train, y_train)

    # Make predictions on the validation data
    val_predictions = decision_tree.predict(X_val)

    # Calculate accuracy or any other evaluation metric
    accuracy = np.mean(val_predictions == y_val)
    validation_scores.append(accuracy)
    print(f"Classification Report - Fold {i}:\n{classification_report(y_val, val_predictions)}")


# Find the index of the fold with the highest validation score
best_fold_index = np.argmax(validation_scores)

# Convert the generator object to a list
kf_splits = list(kf.split(X, y))

# Save the predictions from the best fold to an Excel file
best_fold_predictions = val_predictions
best_fold_predictions_df = pd.DataFrame({'Gözlük kullanıyor mu ?': y.iloc[kf_splits[best_fold_index][1]], 'Predicted value': best_fold_predictions})

# Replace 0 with 'evet' and 1 with 'hayır'
best_fold_predictions_df.replace({0: 'evet', 1: 'hayır'}, inplace=True)

best_fold_predictions_df.to_excel('best_fold_predictions.xlsx', index=False)

print("Predictions from the best fold saved to 'best_fold_predictions.xlsx'")

# Display validation scores for each fold
print("Validation Scores for Each Fold:")
for i, score in enumerate(validation_scores, 1):
    print(f"Fold {i}: {score}")

# Calculate and display the average validation score
average_score = np.mean(validation_scores)
print(f"\nAverage Validation Score: {average_score}")
