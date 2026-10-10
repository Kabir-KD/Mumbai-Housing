import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

print("🔄 Generating dataset and training model...")

# For this survival sprint, we simulate clean Mumbai housing data 
# with core features: Area (sqft), BHK, Bathrooms.
np.random.seed(42)
n_samples = 1000

area = np.random.randint(400, 3500, n_samples)
bhk = np.random.choice([1, 2, 3, 4], n_samples, p=[0.2, 0.4, 0.3, 0.1])
bathrooms = bhk + np.random.choice([0, 1], n_samples, p=[0.8, 0.2])

# Price formula simulation based on Mumbai real estate logic (approx rates per sqft + BHK multiplier)
price = (area * 18000) + (bhk * 2500000) + (bathrooms * 1000000) + np.random.normal(0, 500000, n_samples)
price_in_lakhs = price / 100000  # Convert to Lakhs

# Create DataFrame
df = pd.DataFrame({
    'area': area,
    'bhk': bhk,
    'bathrooms': bathrooms,
    'price': price_in_lakhs
})

# Define Features (X) and Target (y)
X = df[['area', 'bhk', 'bathrooms']]
y = df['price']

# Split into Training and Testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model score
score = model.score(X_test, y_test)
print(f"✅ Model trained successfully! R² Score on Test Data: {score:.2f}")

# Save the trained model to disk
joblib.dump(model, 'mumbai_house_model.pkl')
print("💾 Model saved as 'mumbai_house_model.pkl'")