
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# Load or create synthetic dataset
np.random.seed(42)
n_samples = 200

size = np.random.randint(30, 200, n_samples)
location_score = np.random.randint(1, 11, n_samples)
has_amenities = np.random.randint(0, 2, n_samples)

price = size * 1000 + location_score * 5000 + has_amenities * 10000 + np.random.normal(0, 10000, n_samples)

df = pd.DataFrame({
    'size': size,
    'location_score': location_score,
    'has_amenities': has_amenities,
    'price': price.astype(int)
})

X = df[['size', 'location_score', 'has_amenities']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print("Mean Squared Error:", mse)
