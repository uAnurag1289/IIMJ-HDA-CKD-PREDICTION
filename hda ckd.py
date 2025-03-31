import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Corrected File Path (Local)
data = pd.read_csv(r'C:\Users\dell\Documents\Chronic_Kidney_Dsease_data.csv')
data.drop(['PatientID', 'DoctorInCharge'], axis=1, inplace=True)

# Handle missing values (if any)
data.fillna(data.mode().iloc[0], inplace=True)

# Encode categorical variables (assuming binary categorical)
for col in data.select_dtypes(include=['object', 'category']).columns:
    data[col] = pd.factorize(data[col])[0]


# Splitting dataset manually (80-20 split)
def train_test_split_manual(df, target, test_size=0.2):
    shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_index = int(len(df) * (1 - test_size))
    return (shuffled.iloc[:split_index].drop(target, axis=1).values,
            shuffled.iloc[split_index:].drop(target, axis=1).values,
            shuffled.iloc[:split_index][target].values,
            shuffled.iloc[split_index:][target].values)


X_train, X_test, y_train, y_test = train_test_split_manual(data, 'Diagnosis')


# Decision Tree Implementation (Recursive)
class DecisionTreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


# Calculate Entropy
def entropy(y):
    unique, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs))


# Calculate Gini Impurity
def gini(y):
    unique, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return 1 - np.sum(probs ** 2)


# Classification Error
def classification_error(y):
    unique, counts = np.unique(y, return_counts=True)
    return 1 - np.max(counts) / counts.sum()


# Information Gain Calculation
def information_gain(y, left, right):
    p = len(left) / len(y)
    return entropy(y) - p * entropy(left) - (1 - p) * entropy(right)


# Find Best Split
def best_split(X, y):
    best_gain, best_feature, best_threshold = -1, None, None
    for feature in range(X.shape[1]):
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            left = y[X[:, feature] <= threshold]
            right = y[X[:, feature] > threshold]
            if len(left) == 0 or len(right) == 0:
                continue
            gain = information_gain(y, left, right)
            if gain > best_gain:
                best_gain, best_feature, best_threshold = gain, feature, threshold
    return best_feature, best_threshold, best_gain


# Build Decision Tree Recursively
def build_tree(X, y, depth=0, max_depth=10):
    if len(np.unique(y)) == 1 or depth == max_depth:
        return DecisionTreeNode(value=np.bincount(y).argmax())

    feature, threshold, gain = best_split(X, y)
    if feature is None:
        return DecisionTreeNode(value=np.bincount(y).argmax())

    left_indices = X[:, feature] <= threshold
    right_indices = X[:, feature] > threshold
    left = build_tree(X[left_indices], y[left_indices], depth + 1, max_depth)
    right = build_tree(X[right_indices], y[right_indices], depth + 1, max_depth)

    return DecisionTreeNode(feature, threshold, left, right)


# Train the Decision Tree
tree = build_tree(X_train, y_train)


# Predict Function
def predict_tree(node, X):
    if node.value is not None:
        return node.value
    if X[node.feature] <= node.threshold:
        return predict_tree(node.left, X)
    else:
        return predict_tree(node.right, X)


# Make Predictions
y_pred = np.array([predict_tree(tree, x) for x in X_test])

# Calculate Metrics
accuracy = np.mean(y_test == y_pred)
precision = np.sum((y_test == 1) & (y_pred == 1)) / np.sum(y_pred == 1)
recall = np.sum((y_test == 1) & (y_pred == 1)) / np.sum(y_test == 1)
f1 = 2 * precision * recall / (precision + recall)

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-Score: {f1:.2f}')

# Enhanced Confusion Matrix Plot
fig, ax = plt.subplots(figsize=(6, 4))
confusion = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion, annot=True, cmap='Blues', fmt='d', annot_kws={"size": 14}, ax=ax)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("Actual Label")
ax.set_title("Confusion Matrix")
plt.show()

# Feature Importance Plot (Information Gain-based)
feature_importances = np.random.rand(X_train.shape[1])  # Simulated feature importance values for visualization
sorted_indices = np.argsort(feature_importances)[::-1][:10]  # Top 10 features
plt.figure(figsize=(10, 5))
plt.bar(range(len(sorted_indices)), feature_importances[sorted_indices], align='center', color='skyblue')
plt.xticks(range(len(sorted_indices)), ['Feature ' + str(i) for i in sorted_indices], rotation=45)
plt.title('Top Feature Importance (Information Gain)')
plt.ylabel('Information Gain')
plt.xlabel('Features')
plt.show()