import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

url = "https://halgorithm.com/resources/courses/machine-learning-foundations/used_cars.csv"
df = pd.read_csv(url)

X = df[["year","mileage_km","engine_size_l"]]
y = df["price_usd"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


results = pd.DataFrame({
    "actual": y_test,
    "predicted": y_pred
})

print(results.head())

# Predicting the price of the given datas using the training data.
your_house = pd.DataFrame([[2016, 50000, 1.8]], columns=["year","mileage_km","engine_size_l"])
predicted_price = model.predict(your_house)
print("Predicted price (USD):", predicted_price[0])


print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
