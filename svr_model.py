import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
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


model = SVR(kernel='linear', C=1000)
model.fit(X_train, y_train)


predictions = model.predict(X_test)


svr_mse = mean_squared_error(y_test, predictions)
svr_r2 = r2_score(y_test, predictions)
svr_mae = mean_absolute_error(y_test, predictions)


lot_area = 12500
bedrooms = 4
full_baths = 2
half_baths = 2
garage_cars = 2
overall_condition = 7
year_built = 2002


new_data = {"LotArea": [lot_area], "BedroomAbvGr": [bedrooms], "FullBath": [full_baths], "HalfBath": [half_baths], "OverallCond": [overall_condition], "GarageCars": [garage_cars], "YearBuilt": [year_built]}

new_data_df = pd.DataFrame(new_data)


def predict_svr(new_data_df):
    predicted_price = model.predict(new_data_df)[0]

    discount = calculate_discount(predicted_price)

    final_price = predicted_price - discount
    final_price = round(final_price, 2)

    print("Price after discount : ", "$", final_price)

    return final_price


def calculate_discount(predicted_price):
    discount = 0
    current_year = 2024
    house_age = current_year - new_data_df['YearBuilt'].values[0]

    if house_age >= 10 and new_data_df['OverallCond'].values[0] < 7:
        discount = predicted_price * 0.20
    elif house_age >= 10:
        discount = predicted_price * 0.15
    elif new_data_df['OverallCond'].values[0] < 7:
        discount = predicted_price * 0.05
    return discount

predict_svr(new_data_df)
