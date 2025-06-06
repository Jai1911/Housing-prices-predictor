import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


data = pd.read_csv("train.csv")

features = ["LotArea", "BedroomAbvGr", "FullBath", "HalfBath", "OverallCond", "GarageCars", "YearBuilt"]
target = "SalePrice"


imputer = SimpleImputer(strategy="median")
data[features] = imputer.fit_transform(data[features])


data["TotalBathrooms"] = data["FullBath"] + data["HalfBath"]


X_train, X_test, y_train, y_test = train_test_split(
    data[features], data[target], test_size=0.2, random_state=42
)


model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
model.fit(X_train, y_train)


predictions = model.predict(X_test)


xgb_mse = mean_squared_error(y_test, predictions)
xgb_r2 = r2_score(y_test, predictions)
xgb_mae = mean_absolute_error(y_test, predictions)

#print(f"Mean Squared Error on test set: {mse:.2f}")
#print(f"R-squared on test set: {r2:.2f}")


lot_area = 12500
bedrooms = 4
full_baths = 2
half_baths = 2
garage_cars = 2
overall_condition = 7
year_built = 2002


new_data = {"LotArea": [lot_area], "BedroomAbvGr": [bedrooms], "FullBath": [full_baths], "HalfBath": [half_baths], "OverallCond": [overall_condition], "GarageCars": [garage_cars], "YearBuilt": [year_built]}

new_data_df = pd.DataFrame(new_data)


def predict_xgb(new_data_df):
    predicted_price = model.predict(new_data_df)[0]

    discount = calculate_discount(predicted_price)

    final_price = predicted_price - discount
    final_price = round(final_price, 2)

    #print("Price before discount : ", "$", predicted_price)

    print("Price after discount : ", "$", final_price)

    
    return final_price


def calculate_discount(predicted_price):
  discount = 0
  if new_data_df['YearBuilt'].values[0] >= 10 and new_data_df['OverallCond'].values[0] < 7:
    discount = predicted_price * 0.20
  elif new_data_df['YearBuilt'].values[0] >= 10:
    discount = predicted_price * 0.15
  elif new_data_df['OverallCond'].values[0] < 7:
    discount = predicted_price * 0.05
  return discount


