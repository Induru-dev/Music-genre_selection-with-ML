import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset from the CSV file
df = pd.read_csv("music_preferences.csv")

# Assuming the column names are 'Brightness', 'Weather', 'Temperature', 'Time_of_Day', and 'Music_Genre'
# If the column names are different, adjust accordingly
X = df[['Brightness', 'Weather', 'Temperature', 'Time_of_Day']]
y = df['Music_Genre']

# Encode categorical features
le_weather = LabelEncoder()
le_time_of_day = LabelEncoder()

X['Weather'] = le_weather.fit_transform(X['Weather'])
X['Time_of_Day'] = le_time_of_day.fit_transform(X['Time_of_Day'])

# One-hot encode the encoded features
ohe = OneHotEncoder(sparse_output=False)
X_encoded = ohe.fit_transform(X[['Weather', 'Time_of_Day']])

# Combine numerical and encoded features
X = pd.concat([X[['Brightness', 'Temperature']], pd.DataFrame(X_encoded, columns=ohe.get_feature_names_out())], axis=1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the SVM model
model = SVC(kernel='rbf', C=1.0, gamma='scale',class_weight='balanced')
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Function to recommend music based on user input
def recommend_music(brightness, weather, temperature, time_of_day):
    # Encode the categorical inputs using the previously fitted encoders
    weather_encoded = le_weather.transform([weather])[0]
    time_of_day_encoded = le_time_of_day.transform([time_of_day])[0]
    
    # One-hot encode the encoded categorical inputs
    encoded_features = ohe.transform([[weather_encoded, time_of_day_encoded]])
    
    # Combine the encoded features with the numerical inputs
    user_input = pd.concat([
        pd.DataFrame([[brightness, temperature]], columns=['Brightness', 'Temperature']),
        pd.DataFrame(encoded_features, columns=ohe.get_feature_names_out())
    ], axis=1)
    
    # Predict the music genre
    recommended_genre = model.predict(user_input)
    return recommended_genre[0]

# Example usage of the recommendation function
brightness_input = 70  # Example value for brightness
weather_input = 'Rainy'  # Example value for weather
temperature_input = 22  # Example value for temperature
time_of_day_input = 'Morning'  # Example value for time of day

# Get a music recommendation
recommended_music = recommend_music(brightness_input, weather_input, temperature_input, time_of_day_input)
print(f"Recommended Music Genre: {recommended_music}")


data = {
    'time_of_day': ['Morning', 'Afternoon', 'Evening', 'Night', 'Morning', 'Evening', 'Night', 'Afternoon'],
    'temperature': [20, 30, 25, 15, 28, 22, 19, 31],  # temperatures in °C
    'weather': ['Rain', 'No Rain', 'Rain', 'No Rain', 'Rain', 'No Rain', 'Rain', 'No Rain'],
    'brightness': ['High', 'Low', 'Low', 'High', 'High', 'Low', 'High', 'Low'],
    'music_type': ['Calm', 'Energetic', 'Calm', 'Calm', 'Energetic', 'Calm', 'Calm', 'Energetic']
}

# Hyperparameter Tuning with Grid Search
# Reduce the number of folds to 2 because the data size is very small
#param_grid = {
#    'C': [0.1, 1, 10],
#    'gamma': ['scale', 'auto'],
#    'kernel': ['rbf', 'linear']
#}
#
# Adjust cv parameter to fit the small data size
#grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=2)
#grid.fit(X_train, y_train)
#
# Best parameters found
#print("\nBest Parameters from Grid Search:")
#print(grid.best_params_)
#
# Get the best model from grid search
#best_model = grid.best_estimator_