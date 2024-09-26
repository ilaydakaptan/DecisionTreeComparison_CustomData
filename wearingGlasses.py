# USING THE DECISION TREE ALGORITHM THAT I WRITE

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report
df = pd.read_excel('glassesData.xlsx')

# Replace everything with numeric values
df.replace({'Evet': 0, 'Hayır': 1, 'Kahverengi': 0, 'Diğer': 1,
            'Erkek': 0, 'Kadın': 1, '15 - 30': 0, '31 - 45': 1,
            '> 45': 2, '< 15': 3, '<3': 0, '3 - 6': 1, '>6': 2}, inplace=True)


# Save the modified data to a new Excel file
df.to_excel('modified_glassesData.xlsx', index=False)

class Node:
    def __init__(self, feature=None, threshold=None, value=None, left=None, right=None, result=None):
        self.feature = feature
        self.threshold = threshold
        self.value = value
        self.left = left
        self.right = right
        self.result = result

# To keep track of the node count
Node.node_count = 0

def entropy(labels):
    unique_labels, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))

def information_gain(data, feature, threshold, target):
    total_entropy = entropy(data[target])

    left_mask = data[feature] <= threshold
    right_mask = ~left_mask

    left_entropy = entropy(data[left_mask][target])
    right_entropy = entropy(data[right_mask][target])

    left_weight = len(data[left_mask]) / len(data)
    right_weight = len(data[right_mask]) / len(data)

    gain = total_entropy - (left_weight * left_entropy + right_weight * right_entropy)
    return gain

def find_best_split(data, features, target):
    best_gain = -1
    best_feature = None
    best_threshold = None

    for feature in features:
        unique_values = data[feature].unique()
        for value in unique_values:
            gain = information_gain(data, feature, value, target)

            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = value

    return best_feature, best_threshold

def build_decision_tree(data, features, target):
    Node.node_count += 1  # Increment node count when a new node is created

    if len(np.unique(data[target])) == 1:
        return Node(result=data[target].iloc[0])

    if len(features) == 0:
        majority_label = data[target].mode().iloc[0]
        return Node(result=majority_label)

    best_feature, best_threshold = find_best_split(data, features, target)

    if best_feature is None:
        majority_label = data[target].mode().iloc[0]
        return Node(result=majority_label)

    left_mask = data[best_feature] <= best_threshold
    right_mask = ~left_mask

    # Check if the left or right subtree is empty
    if len(data[left_mask]) == 0 or len(data[right_mask]) == 0:
        majority_label = data[target].mode().iloc[0]
        return Node(result=majority_label)

    left_subtree = build_decision_tree(data[left_mask], features, target)
    right_subtree = build_decision_tree(data[right_mask], features, target)

    return Node(feature=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)

def predict(tree, example):
    if tree.result is not None:
        return tree.result

    if example[tree.feature] <= tree.threshold:
        return predict(tree.left, example)
    else:
        return predict(tree.right, example)

# Load the entire dataset
full_data = pd.read_excel("modified_glassesData.xlsx")

# Separate features and target variable
X = full_data.iloc[:, :-1]
y = full_data.iloc[:, -1]

# Initialize StratifiedKFold for 10-fold cross-validation
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Create an empty list to store validation scores
validation_scores = []

# Perform 10-fold cross-validation
for i, (train_index, test_index) in enumerate(kf.split(X, y), 1):
    X_train, X_val = X.iloc[train_index], X.iloc[test_index]
    y_train, y_val = y.iloc[train_index], y.iloc[test_index]

    # Build the decision tree using the training data
    features = X_train.columns.tolist()
    target = y_train.name
    decision_tree = build_decision_tree(pd.concat([X_train, y_train], axis=1), features, target)

    # Make predictions on the validation data
    val_predictions = [predict(decision_tree, example) for _, example in X_val.iterrows()]

    # Calculate accuracy or any other evaluation metric
    accuracy = np.mean(val_predictions == y_val)
    validation_scores.append(accuracy)
    print(f"Classification Report - Fold {i}:\n{classification_report(y_val, val_predictions)}")


# Display the tree structure and the number of nodes
def display_tree_structure(node, depth=0):
    if node.result is not None:
        print(f"{'  ' * depth}Result: {node.result}")
    else:
        print(f"{'  ' * depth}{node.feature} <= {node.threshold}")
        display_tree_structure(node.left, depth + 1)
        display_tree_structure(node.right, depth + 1)


print("Decision Tree Structure:")
display_tree_structure(decision_tree)
print(f"Number of Nodes: {Node.node_count}")

# Display validation scores for each fold
print("Validation Scores for Each Fold:")
for i, score in enumerate(validation_scores, 1):
    print(f"Fold {i}: {score}")

# Calculate and display the average validation score
average_score = np.mean(validation_scores)
print(f"\nAverage Validation Score: {average_score}")




