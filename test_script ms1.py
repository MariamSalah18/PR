import pickle
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score
import warnings
from datetime import datetime
from sklearn import metrics
from sklearn.metrics import r2_score, mean_squared_error
warnings.filterwarnings('ignore')

default_value="unknown"
def fill_mode(x):
    if not x.mode().empty:
        return x.mode().iloc[0]  # Use mode if available
    else:
       return default_value

def set_category(text):
    text_lower = text.lower()  # Convert text to lowercase
    if 'rent' in text_lower:
        return 'rent'
    elif 'apartment' in text_lower:
        return 'apartment'
    else:
        return 'housing'

mean_values_loaded = pickle.load(open('mean_values.pkl', 'rb'))
mode_values_loaded = pickle.load(open('mode_values.pkl', 'rb'))
mode_rent_loaded = pickle.load(open('mode_rent.pkl', 'rb'))
mode_apartment_loaded = pickle.load(open('mode_apartment.pkl', 'rb'))
mode_housing_loaded = pickle.load(open('mode_housing.pkl', 'rb'))

test_data = pd.read_csv("ApartmentRentPrediction_test.csv")

selected_features = ['category','body','amenities', 'bathrooms','bedrooms','has_photo','pets_allowed',
                   'square_feet','address','cityname','state','latitude','longitude','source','time']
X = test_data.drop(columns=['price_display'])
Y = test_data['price_display']
# Remove non-numeric characters and convert to integers
Y = Y.str.replace('[^\d]', '', regex=True).astype(int)

X_columns = X.columns
# Using dictionary comprehension to create the dictionary
mean_dict = {column_name: mean_value for column_name, mean_value in zip(X_columns, mean_values_loaded)}
mode_dict = {column_name: mode_value for column_name, mode_value in zip(X_columns, mode_values_loaded)}

X = X[selected_features]
X['category'] = X['body'].apply(set_category)
X.loc[X['category'] == 'rent', 'amenities']=X.loc[X['category'] == 'rent', 'amenities'].fillna(mode_rent_loaded)
X.loc[X['category'] == 'apartment', 'amenities']=X.loc[X['category'] == 'apartment', 'amenities'].fillna(mode_apartment_loaded)
X.loc[X['category'] == 'housing', 'amenities']=X.loc[X['category'] == 'housing', 'amenities'].fillna(mode_housing_loaded)
X['pets_allowed'].fillna('None', inplace=True)
X['source'] = X['source'].apply(lambda x: x + '.com' if not x.endswith('.com') else x)

X['time'] = [datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') for ts in X['time']]

X['bathrooms'] = X['bathrooms'].fillna(mode_dict['bathrooms']).astype(int)
X['bedrooms'].replace(0, 1, inplace=True)
X['bedrooms'] = X['bedrooms'].fillna(mode_dict['bedrooms'])
X['cityname'] = X['cityname'].fillna(mode_dict['cityname'])
X['state'] = X['state'].fillna(mode_dict['state'])
X['longitude'] = X['longitude'].fillna(mean_dict['longitude'])
X['latitude'] = X['latitude'].fillna(mean_dict['latitude'])

X['address'] = X.groupby('cityname')['address'].transform(fill_mode)
X = X.drop(columns=['body'])
label_encoders = {}
scalers = {}
categorical_columns = X.select_dtypes(include=['object']).columns
for column in categorical_columns:
    # Load the LabelEncoder object
    with open(f"label_encoder_{column}.pkl", "rb") as file:
        label_encoder = pickle.load(file)
    # Store the LabelEncoder object in the dictionary
    label_encoders[column] = label_encoder
    # Get unique labels from the column
    unique_labels = set(X[column])
    # Check for unseen labels
    unseen_labels = unique_labels - set(label_encoder.classes_)
    # If there are unseen labels, append them to the label encoder classes
    if unseen_labels:
        for label in unseen_labels:
            label_encoder.classes_ = np.append(label_encoder.classes_, label)

    # Transform the corresponding column in X
    X[column] = label_encoder.transform(X[column])

threshold = 3

# Iterate over each column
for column in X.columns:
    # Calculate the z-score for each value in the column
    z_scores = (X[column] - X[column].mean()) / X[column].std()

    # Find outliers (values with z-score greater than the threshold)
    outliers = X[abs(z_scores) > threshold][column]

    # Replace outliers with mean values from result_dict
    X.loc[outliers.index, column] = mean_dict[column]

scalers = {}
for column in X.columns:
    with open(f"normalized_values_{column}.pkl", "rb") as file:
        # Load the scaler object
        scaler = pickle.load(file)

    # Store the scaler object in the dictionary
    scalers[column] = scaler

    # Transform the corresponding column in X using the loaded scaler
    X[column] = scaler.transform(X[column].values.reshape(-1, 1))

linear_model_loaded = pickle.load(open('linear_reg.pkl', 'rb'))
poly_model_loaded = pickle.load(open('poly_reg.pkl', 'rb'))
rf_model_loaded = pickle.load(open('random_forest.pkl', 'rb'))
svr_model_loaded = pickle.load(open('svr.pkl', 'rb'))
gb_model_loaded = pickle.load(open('gradient_boosting.pkl', 'rb'))
poly_features_loaded = pickle.load(open('poly_features.pkl', 'rb'))

# Make predictions using the loaded model
predictions_linear = linear_model_loaded.predict(X)
X = poly_features_loaded.transform(X)
predictions_poly = poly_model_loaded.predict(X)
predictions_rf = rf_model_loaded.predict(X)
predictions_gb = gb_model_loaded.predict(X)
predictions_svr = svr_model_loaded.predict(X)

linear_r2 = r2_score(Y, predictions_linear)
print('Mean Square Error linear', metrics.mean_squared_error(Y, predictions_linear))
print("Linear Regression R-squared score:", linear_r2*100,'%')

poly_r2 = r2_score(Y, predictions_poly)
print('Mean Square Error Polynomial', metrics.mean_squared_error(Y, predictions_poly))
print("polynomial Regression R-squared score:", poly_r2*100,'%')

rf_r2 = r2_score(Y, predictions_rf)
print('Mean Square Error random forest', metrics.mean_squared_error(Y, predictions_rf))
print("random forest Regression R-squared score:", rf_r2*100,'%')

gb_r2 = r2_score(Y, predictions_gb)
print('Mean Square Error gradient boosting', metrics.mean_squared_error(Y, predictions_gb))
print("gradient boosting Regression R-squared score:", gb_r2*100,'%')

svr_r2 = r2_score(Y, predictions_svr)
print('Mean Square Error SVR', metrics.mean_squared_error(Y, predictions_svr))
print("svr Regression R-squared score:", svr_r2*100,'%')
