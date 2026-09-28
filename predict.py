import joblib
import pandas as pd


# Load trained model
model = joblib.load("models/crop_model.pkl")


# User input
input_data = pd.DataFrame([{
    "N": 90,
    "P": 42,
    "K": 43,
    "temperature": 25,
    "humidity": 80,
    "ph": 6.5,
    "rainfall": 200
}])


# Get probabilities
probabilities = model.predict_proba(input_data)[0]

# Get crop names
crop_names = model.classes_

# Create crop-probability pairs
results = list(zip(crop_names, probabilities))

# Sort by probability
results.sort(key=lambda x: x[1], reverse=True)


print("\n🌾 Crop Recommendations")
print("------------------------")

for crop, probability in results[:3]:
    print(f"{crop}: {probability * 100:.2f}%")