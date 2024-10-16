# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Load dataset
df = pd.read_csv('music_recommendation_100k_more_acc.csv')

# Preprocessing
label_encoders = {}
for column in ['time_of_day', 'weather', 'brightness', 'music_type']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Split features and target variable
X = df[['time_of_day', 'temperature', 'weather', 'brightness']]
y = df['music_type']

# Scale the numerical 'temperature' feature
scaler = StandardScaler()
X['temperature'] = scaler.fit_transform(X[['temperature']]).astype(float)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train the SVM model
model = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced')
model.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(model, 'svm_music_recommendation_model.pkl')
print("Model trained and saved successfully as 'svm_music_recommendation_model.pkl'")

# Save the encoders and scaler to ensure consistent preprocessing on the Raspberry Pi
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Encoders and scaler saved successfully.")

# Evaluate the model on the test set
y_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=1))
