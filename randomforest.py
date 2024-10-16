# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score

# Load dataset
df = pd.read_csv('music_100_test1.csv')

# Preprocessing
# Label Encoding categorical features
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

# Check the sizes of training and test sets to avoid splits-related errors
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# Define the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# Train the model
model.fit(X_train, y_train)

# Cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print("Cross-Validation Accuracy Scores:", cv_scores)
print("Mean Cross-Validation Accuracy:", cv_scores.mean())

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=1))

# Calculate accuracy on the test set
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Function to predict music type based on input
def recommend_music(time_of_day, temperature, weather, brightness):
    # Convert input to match the encoding used during training
    encoded_time_of_day = label_encoders['time_of_day'].transform([time_of_day])[0]
    encoded_weather = label_encoders['weather'].transform([weather])[0]
    encoded_brightness = label_encoders['brightness'].transform([brightness])[0]
    
    # Scale the temperature input using the same scaler
    scaled_temperature = scaler.transform([[temperature]])[0][0]
    
    # Create input array for the model with proper feature names
    input_features = pd.DataFrame([[encoded_time_of_day, scaled_temperature, encoded_weather, encoded_brightness]],
                                  columns=['time_of_day', 'temperature', 'weather', 'brightness'])
    
    # Predict music type using the Random Forest model
    prediction = model.predict(input_features)
    
    # Convert the prediction back to the original label
    recommended_music = label_encoders['music_type'].inverse_transform(prediction)[0]
    
    return recommended_music

# Example usage: replace these values with the actual input
user_time_of_day = 'Morning'  # Example: Morning, Afternoon, Evening, Night
user_temperature = 37         # Example: temperature in °C
user_weather = 'No Rain'      # Example: Rain, No Rain
user_brightness = 'High'      # Example: High, Low

# Get the music recommendation
recommendation = recommend_music(user_time_of_day, user_temperature, user_weather, user_brightness)
print(f"\nRecommended Music Type: {recommendation}")
