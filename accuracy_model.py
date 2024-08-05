import pandas as pd

# Import functions from model files (replace with actual file paths and function names)
from random_forest import predict_rf, rf_r2, rf_mae  # Random Forest prediction function
from svr_model import predict_svr, svr_r2, svr_mae  # SVR prediction function
from xgboost_model import predict_xgb, xgb_r2, xgb_mae  # XGBoost prediction function

def get_user_data():

    lot_area = float(input("Enter the lot area (in square feet): "))
    bedrooms = int(input("Enter the number of bedrooms: "))
    full_baths = int(input("Enter the number of full bathrooms: "))
    half_baths = int(input("Enter the number of half bathrooms: "))
    garage_cars = int(input("Enter the number of garage cars: "))
    overall_condition = int(input("Enter the overall condition (1-10): "))
    year_built = int(input("Enter the year in which the house was constructed: "))

    return {
        "LotArea": lot_area,
        "BedroomAbvGr": bedrooms,
        "FullBath": full_baths,
        "HalfBath": half_baths,
        "OverallCond": overall_condition,
        "GarageCars": garage_cars,
        "YearBuilt": year_built
    }


def create_data_from_user_input():

  user_data = get_user_data()
  return pd.DataFrame([user_data])  # Create a DataFrame with a single row

# Get user input for the new house
new_data_df = create_data_from_user_input()

# Split the user-provided data into features (X_new) for prediction
features = ["LotArea", "BedroomAbvGr", "FullBath", "HalfBath", "OverallCond", "GarageCars", "YearBuilt"]
X_new = new_data_df[features]

# Make predictions using each model function
rf_predicted_price = predict_rf(X_new)
svr_predicted_price = predict_svr(X_new)
xgb_predicted_price = predict_xgb(X_new)

# Print the predicted prices for the user-provided house
print("Final rices for the new house, after discount:")
print(f"Random Forest: ${rf_predicted_price:.2f}")
print(f"SVR: ${svr_predicted_price:.2f}")
print(f"XGBoost: ${xgb_predicted_price:.2f}")

if rf_mae < svr_mae and rf_mae < xgb_mae:
    print("Random Forest give the most accurate price : ", rf_predicted_price)
elif svr_mae < rf_mae and svr_mae < xgb_mae:
    print("SVR gives the most accurate price : ", svr_predicted_price)
else:
    print("XGBoost gives the most accurate price : ", xgb_predicted_price)
