import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv("booking.csv")

# Drop ID column
df = df.drop("Booking_ID", axis=1)

# Fix invalid dates
df["date of reservation"] = pd.to_datetime(df["date of reservation"], errors="coerce")

# Remove invalid rows
df = df.dropna()

# Extract month and day
df["reservation_month"] = df["date of reservation"].dt.month
df["reservation_day"] = df["date of reservation"].dt.day

df = df.drop("date of reservation", axis=1)

# Encode categorical columns
categorical_cols = ["type of meal", "room type", "market segment type"]

encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Encode target
target_encoder = LabelEncoder()
df["booking status"] = target_encoder.fit_transform(df["booking status"])

# Split data
X = df.drop("booking status", axis=1)
y = df["booking status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

# Save files
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(encoders, open("encoders.pkl", "wb"))
pickle.dump(target_encoder, open("target_encoder.pkl", "wb"))

print("Model trained successfully!")