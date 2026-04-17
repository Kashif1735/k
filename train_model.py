import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error

print("Loading Bengaluru house data...")
df = pd.read_csv('Bengaluru_House_Data.csv')
print(f"Shape: {df.shape}")

# Remove unwanted columns
df = df.drop(['area_type', 'availability', 'society', 'balcony'], axis=1)

# Fill missing
df["location"] = df["location"].fillna("Sarjapur Road")
df["size"] = df["size"].fillna("2 BHK")
med_bath = df["bath"].median()
df["bath"] = df["bath"].fillna(med_bath)
df["bath"] = df["bath"].astype(int)
df.drop_duplicates(inplace=True)

# Location cleaning
df['location'] = df['location'].apply(lambda x: x.strip())
loc_counts = df['location'].value_counts()
loc_less10 = loc_counts[loc_counts <= 10]
df['location'] = df['location'].apply(lambda x: 'others' if x in loc_less10 else x)

# BHK from size
df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]))
df.drop('size', axis=1, inplace=True)

# Total sqft
def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None

df['total_sqft'] = df['total_sqft'].apply(convert_sqft_to_num)
df['total_sqft'] = df['total_sqft'].fillna(df['total_sqft'].mean())

# Price per sqft for outliers
df['price_per_sqft'] = df['price'] * 100000 / df['total_sqft']

# Outliers
df = df[df['total_sqft']/df['bhk'] >= 300]
df = df[df['bhk'] <= 6]
df = df[df['bath'] < df['bhk'] + 2]

q1 = df['price_per_sqft'].quantile(0.25)
q3 = df['price_per_sqft'].quantile(0.75)
iqr = q3 - q1
lower = q1 - 0.5 * iqr
upper = q3 + 0.5 * iqr
df = df[(df['price_per_sqft'] >= lower) & (df['price_per_sqft'] <= upper)]

df.reset_index(drop=True, inplace=True)
df.drop(['price_per_sqft'], axis=1, inplace=True)

print("Encoding locations...")
df = pd.get_dummies(df, columns=['location'], drop_first=True, dtype=int)

# Features and target
X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Training RandomForest with GridSearch...")
model = RandomForestRegressor(random_state=42)
params = {
    'n_estimators': [100, 150, 200],
    'max_depth': [3, 4, 5, 6, 7]
}
grid = GridSearchCV(model, params, cv=5)
grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
print("Best CV score:", grid.best_score_)

model = grid.best_estimator_

# Test
y_pred = model.predict(X_test)
print("Test R2:", r2_score(y_test, y_pred))
print("Test MAE:", mean_absolute_error(y_test, y_pred))

# Save
df.to_csv('cleaned_df.csv', index=False)
joblib.dump(model, 'rf_model.joblib')
joblib.dump(X.columns.tolist(), 'model_columns.joblib')

print("Model and data saved! Ready for app.")

