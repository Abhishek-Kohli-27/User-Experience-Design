import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('user_interaction_data.csv')

# Preview the data
data.head()


# Check for missing values
print(data.isnull().sum())

# Drop or fill missing values
data = data.dropna()

# Identify categorical columns
categorical_cols = data.select_dtypes(include=['object']).columns

# Encode categorical variables
label_encoder = LabelEncoder()
for col in categorical_cols:
    data[col] = label_encoder.fit_transform(data[col])

# Identify numerical columns
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns

# Scale numerical features
scaler = StandardScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Correlation matrix
corr_matrix = data.corr()

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f')
plt.show()

# Select features with high correlation to the target variable
correlation_threshold = 0.1
relevant_features = corr_matrix['target_variable'][abs(corr_matrix['target_variable']) > correlation_threshold].index
print("Relevant features:", relevant_features)

# Define features and target
X = data[relevant_features].drop('target_variable', axis=1)
y = data['target_variable']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Classification report
print(classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.show()

# Feature importance
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances.sort_values(ascending=False, inplace=True)

# Plot feature importance
feature_importances.plot(kind='bar')
plt.title('Feature Importance')
plt.show()

# Relationship between key features and target variable
sns.pairplot(data, vars=feature_importances.index[:5], hue='target_variable')
plt.show()

