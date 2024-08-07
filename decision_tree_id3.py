import pandas as pd
import numpy as np


def read_csv(file_path):
    return pd.read_csv(file_path)

def strip_column_names(df):
    """Strip leading and trailing spaces from column names"""
    df.columns = df.columns.str.strip()
    return df

def entropy(y):
    """Compute the entropy of a label array"""
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities))

def information_gain(X_column, y, split_value):
    """Compute the information gain of a split"""
    parent_entropy = entropy(y)
    
    left_mask = X_column <= split_value
    right_mask = X_column > split_value
    
    if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
        return 0

    left_entropy = entropy(y[left_mask])
    right_entropy = entropy(y[right_mask])

    weighted_avg_entropy = (len(y[left_mask]) / len(y) * left_entropy +
                            len(y[right_mask]) / len(y) * right_entropy)
    
    return parent_entropy - weighted_avg_entropy

def best_split(X, y):
    """Find the best split for a dataset"""
    best_gain = 0
    best_split_value = None
    best_split_column = None
    
    for column in X.columns:
        split_values = X[column].unique()
        for split_value in split_values:
            gain = information_gain(X[column], y, split_value)
            if gain > best_gain:
                best_gain = gain
                best_split_value = split_value
                best_split_column = column
    
    return best_split_column, best_split_value

def id3(X, y, depth=0, max_depth=None):
    """Build the decision tree using the ID3 algorithm"""
    # If only one class left or no features left, return the class label
    if len(np.unique(y)) == 1:
        return y.iloc[0]
    if X.empty or (max_depth and depth == max_depth):
        return y.mode()[0]
    
    # Find the best split
    best_column, best_value = best_split(X, y)
    if best_column is None:
        return y.mode()[0]
    
    # Create subtrees
    tree = {best_column: {}}
    left_mask = X[best_column] <= best_value
    right_mask = X[best_column] > best_value
    
    tree[best_column]['<=' + str(best_value)] = id3(X[left_mask], y[left_mask], depth + 1, max_depth)
    tree[best_column]['>' + str(best_value)] = id3(X[right_mask], y[right_mask], depth + 1, max_depth)
    
    return tree

def predict(tree, sample):
    """Predict the class label for a new sample"""
    if not isinstance(tree, dict):
        return tree
    
    feature = list(tree.keys())[0]
    value = sample[feature]
    
    for branch_value, subtree in tree[feature].items():
        operator, split_value = branch_value[0], branch_value[1:]
        if (operator == '<=' and value <= float(split_value)) or \
           (operator == '>' and value > float(split_value)):
            return predict(subtree, sample)

# Example usage:
file_path = 'decision_tree_id3.csv'
data = read_csv(file_path)

# Strip extra spaces from column names
data = strip_column_names(data)
# print("Columns in the dataset:", data.columns)

X = data.drop('Class', axis=1)
y = data['Class']

tree = id3(X, y)
print("Decision Tree:")
print(tree)

# Classify a new sample
new_sample = pd.Series([5.1, 3.5, 1.4, 0.2], index=X.columns)
print("Class for new sample:", predict(tree, new_sample))
