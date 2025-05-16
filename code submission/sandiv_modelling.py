import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier # For KNN
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Configuration
MODELING_DATA_PATH = 'data_for_modeling.csv'

# Load Processed Data for Modeling
try:
    # Load the data specifically prepared for modeling
    accident_model_df = pd.read_csv(MODELING_DATA_PATH)
    print(f"Modeling data loaded from '{MODELING_DATA_PATH}': {accident_model_df.shape[0]} rows, {accident_model_df.shape[1]} columns")
except FileNotFoundError:
    print(f"CRITICAL ERROR: Modeling data file '{MODELING_DATA_PATH}' not found.")
    print("Please ensure 'toby_script.py' has been run and saved 'data_for_modeling.csv'.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the modeling CSV: {e}")
    exit()


features_for_model = [
    'SPEED_ZONE_GROUP',
    'LIGHT_CONDITION_DESC',
    'DAY_WEEK_DESC',
    'ROAD_GEOMETRY_DESC',
    'NO_OF_VEHICLES'
]
target_variable = 'IS_SERIOUS_FATAL'

# Check if target and all selected features are present
if target_variable not in accident_model_df.columns:
    print(f"CRITICAL ERROR: Target variable '{target_variable}' is missing from {MODELING_DATA_PATH}")
    exit()
missing_features = [f for f in features_for_model if f not in accident_model_df.columns]
if missing_features:
    print(f"CRITICAL ERROR: The following features are missing from {MODELING_DATA_PATH}: {missing_features}")
    print(f"Available columns in loaded modeling data: {accident_model_df.columns.tolist()}")
    exit()

X = accident_model_df[features_for_model]
y = accident_model_df[target_variable]

# Handle Missing Values
for col in X.select_dtypes(include=np.number).columns:
    if X[col].isnull().any():
        X.loc[:, col] = X[col].fillna(X[col].median())

for col in X.select_dtypes(include='object').columns:
    if X[col].isnull().any():
         X.loc[:, col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown')


# Define Preprocessing
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()

# Create the preprocessor
# KNN requires all features to be numerical and scaled.
# OneHotEncoder makes all features numerical, which is good for KNN.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
    ]
)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Training data shape: X_train {X_train.shape}, y_train {y_train.shape}")
print(f"Testing data shape: X_test {X_test.shape}, y_test {y_test.shape}")
print(f"Class distribution in training target: \n{y_train.value_counts(normalize=True)}")
print(f"Class distribution in testing target: \n{y_test.value_counts(normalize=True)}")


# Train, Predict, Evaluate Models

# Model 1: Decision Tree Classifier
print("\n--- Decision Tree Classifier ---")
dt_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', DecisionTreeClassifier(random_state=42, class_weight='balanced', max_depth=5))])
dt_pipeline.fit(X_train, y_train)
y_pred_dt = dt_pipeline.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Decision Tree Classification Report:\n", classification_report(y_test, y_pred_dt, zero_division=0))
print("Decision Tree Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))

# Model 2: K-Nearest Neighbors (KNN) Classifier
print("\n--- K-Nearest Neighbors (KNN) Classifier ---")
knn_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', KNeighborsClassifier(n_neighbors=5))])
knn_pipeline.fit(X_train, y_train)
y_pred_knn = knn_pipeline.predict(X_test)

print("KNN Accuracy (k=5):", accuracy_score(y_test, y_pred_knn))
print("KNN Classification Report (k=5):\n", classification_report(y_test, y_pred_knn, zero_division=0))
print("KNN Confusion Matrix (k=5):\n", confusion_matrix(y_test, y_pred_knn))

# Below code is from Gemini 2.5 Pro
# Compare Models 
print("\n--- Model Comparison Summary ---")
report_dt = classification_report(y_test, y_pred_dt, output_dict=True, zero_division=0)
report_knn = classification_report(y_test, y_pred_knn, output_dict=True, zero_division=0)

print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
# Check if '1' (for positive class) exists in report before accessing
if '1' in report_dt:
    print(f"Decision Tree F1-score (Serious/Fatal): {report_dt['1']['f1-score']:.4f}")
    print(f"Decision Tree Precision (Serious/Fatal): {report_dt['1']['precision']:.4f}")
    print(f"Decision Tree Recall (Serious/Fatal): {report_dt['1']['recall']:.4f}")
else:
    print("Decision Tree: Class '1' (Serious/Fatal) not present in test predictions or actuals to calculate F1/Precision/Recall.")


print(f"\nKNN Accuracy (k=5): {accuracy_score(y_test, y_pred_knn):.4f}")
if '1' in report_knn:
    print(f"KNN F1-score (Serious/Fatal - k=5): {report_knn['1']['f1-score']:.4f}")
    print(f"KNN Precision (Serious/Fatal - k=5): {report_knn['1']['precision']:.4f}")
    print(f"KNN Recall (Serious/Fatal - k=5): {report_knn['1']['recall']:.4f}")
else:
    print("KNN: Class '1' (Serious/Fatal) not present in test predictions or actuals to calculate F1/Precision/Recall.")

