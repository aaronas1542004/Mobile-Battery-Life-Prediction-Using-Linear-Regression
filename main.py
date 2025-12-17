import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

df = pd.read_csv('battery_life.csv')
print("Dataset Preview:")
print(df.head())
df['network'] = df['network'].map({'WiFi': 0, 'MobileData': 1})

X = df[['screen_time', 'apps_used', 'network']]
y = df['battery_hours']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)
print("\nModel training complete.")


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nEvaluation Metrics:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")


example_usage = [[5, 20, 0]]
predicted_hours = model.predict(example_usage)
print(f"\nPredicted battery life for 5h screen time on WiFi: {predicted_hours[0]:.2f} hours")


C:\Users\Student\PycharmProjects\BatteryLifePrediction\.venv\Scripts\python.exe C:\Users\Student\PycharmProjects\BatteryLifePrediction\main.py 
Dataset Preview:
   screen_time  apps_used     network  battery_hours
0            2         10        WiFi             18
1            3         15        WiFi             16
2            4         20  MobileData             14
3            5         25  MobileData             12
4            6         30  MobileData             10

Model training complete.

Evaluation Metrics:
Mean Squared Error: 0.69
R-squared Score: 0.94

Predicted battery life for 5h screen time on WiFi: 15.77 hours
