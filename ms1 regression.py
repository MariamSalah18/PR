import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
import warnings
import pickle
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from itertools import combinations
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from scipy import stats
from datetime import datetime
from geopy.geocoders import Nominatim
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
warnings.filterwarnings('ignore')

df = pd.read_csv('ApartmentRentPrediction.csv')
X = df.drop(columns=['price_display'])
Y = df['price_display']
# Remove non-numeric characters and convert to integers
Y = Y.str.replace('[^\d]', '', regex=True).astype(int)
label_encoder=0
scaler=0

#colunms_names=['bathrooms','bedrooms','category','amenities','cityname','state','longitude','latitude']
# Lists to store mean and mode
mean_values = []
mode_values = []
#label_encoders = {}

def set_category(text):
    text_lower = text.lower()  # Convert text to lowercase
    if 'rent' in text_lower:
        return 'rent'
    elif 'apartment' in text_lower:
        return 'apartment'
    else:
        return 'housing'

default_value="unknown"
def fill_mode(x):
    if not x.mode().empty:
        return x.mode().iloc[0]  # Use mode if available
    else:
       return default_value

def replace_outliers_with_mean(df):
    z_scores = np.abs(stats.zscore(df))
    threshold = 3
    is_outlier = z_scores > threshold
    column_means = df.mean()
    df_no_outliers = df.mask(is_outlier, column_means, axis=1)
    return df_no_outliers

X['category'] = X['body'].apply(set_category)
X['source'] = X['source'].apply(lambda x: x + '.com' if not x.endswith('.com') else x)
# Convert each timestamp to datetime
X['time'] = [datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') for ts in X['time']]
X['bathrooms'].replace([np.inf, -np.inf], 1, inplace=True)
X['bathrooms'].replace(0, 1, inplace=True)
# Fill missing values with the mode
X['bathrooms'] = X['bathrooms'].fillna(X['bathrooms'].mode()[0])
# Convert to integer
X['bathrooms'] = X['bathrooms'].astype(int)
X['bedrooms'].replace(0, 1, inplace=True)
# Fill missing values with the mode
X['bedrooms'] = X['bedrooms'].fillna(X['bedrooms'].mode()[0])
# Convert to integer
X['bedrooms'] = X['bedrooms'].astype(int)
# Calculate mode values for 'amenities' for each category
mode_rent = X[X['category'] == 'rent']['amenities'].mode()[0]
mode_apartment = X[X['category'] == 'apartment']['amenities'].mode()[0]
mode_housing = X[X['category'] == 'housing']['amenities'].mode()[0]
# Fill NaN values in 'amenities' based on category
X.loc[X['category'] == 'rent', 'amenities'] = X.loc[X['category'] == 'rent', 'amenities'].fillna(mode_rent)
X.loc[X['category'] == 'apartment', 'amenities'] = X.loc[X['category'] == 'apartment', 'amenities'].fillna(mode_apartment)
X.loc[X['category'] == 'housing', 'amenities'] = X.loc[X['category'] == 'housing', 'amenities'].fillna(mode_housing)
X['pets_allowed'].fillna('None', inplace=True)
X['cityname'].fillna(X['cityname'].mode()[0], inplace=True)
# Fill missing values in 'state' with mode
X['state'].fillna(X['state'].mode()[0], inplace=True)
# Apply the function to fill missing addresses based on mode addresses or single addresses for each city
X['address'] = X.groupby('cityname')['address'].transform(fill_mode)
X['longitude'].fillna(X['longitude'].mean(), inplace=True)
X['latitude'].fillna(X['latitude'].mean(), inplace=True)

#saving encoding values
# Identify columns with object data type
categorical_columns = X.select_dtypes(include=['object']).columns
for column in categorical_columns:
    label_encoder = LabelEncoder()
    X[column] = label_encoder.fit_transform(X[column])
    # Save the LabelEncoder object
    with open(f"label_encoder_{column}.pkl", "wb") as file:
        pickle.dump(label_encoder, file)

#saving mean & mode
for col in X.columns:
    # Fill missing values with the mean and save mean
    mean_val = X[col].mean()
    mean_values.append(mean_val)
    X[col].fillna(mean_val, inplace=True)
    # Fill missing values with the mode and save mode
    mode_val = X[col].mode()[0]
    mode_values.append(mode_val)
    X[col] = X[col].fillna(mode_val)

# Concatenate data
data = pd.concat([X, Y], axis=1)
# Replace outliers
data_no_outliers = replace_outliers_with_mean(data)
# Split data with replaced outliers
X_no_outliers = data_no_outliers.drop(columns=['price_display'])
Y_no_outliers = data_no_outliers['price_display']

columns_to_normalize = ['category', 'amenities', 'bathrooms','bedrooms'
    ,'has_photo','pets_allowed','price','square_feet','address','cityname','state','latitude','longitude','source','time']
for column in columns_to_normalize:
    scaler = MinMaxScaler()
    column_values = X_no_outliers[column].values.reshape(-1, 1)
    normalized_values = scaler.fit_transform(column_values)
    X_no_outliers[column] = normalized_values
    with open(f"normalized_values_{column}.pkl", "wb") as file:
        pickle.dump(scaler, file)

columns_to_standardize = ['title', 'body']
scaler = StandardScaler()
X_no_outliers[columns_to_standardize] = scaler.fit_transform(X_no_outliers[columns_to_standardize])

# Features selection
# Drop some columns
columns_to_drop = ['body', 'price', 'id', 'currency', 'fee', 'price_type', 'title']
data_no_outliers.drop(columns=columns_to_drop, inplace=True)
X_no_outliers.drop(columns=columns_to_drop, inplace=True)

# Calculate correlation matrix
corr_matrix = data_no_outliers.corr()

# Features selection using correlation
# top_feature = corr_matrix.index[abs(corr_matrix['price_display']) >= 0.1]
# top_feature = top_feature.delete(-1)
# X_new = X_no_outliers[top_feature]
# X_new['latitude'] = X_no_outliers['latitude']
# X_new['amenities'] = X_no_outliers['amenities']
# print(X_new)
# Plot heatmap
# plt.figure(figsize=(15, 8))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
# plt.title('Correlation Heatmap')
# plt.show()


X_train, X_test, y_train, y_test = train_test_split(X_no_outliers, Y_no_outliers, test_size= 0.20, shuffle=True, random_state=10)

#Linear Regression
model = linear_model.LinearRegression()
model.fit(X_train, y_train)
prediction = model.predict(X_test)
print('Mean Square Error Linear', metrics.mean_squared_error(y_test, prediction))
# Calculate R-squared score
r2 = r2_score(y_test, prediction)
print('R2 score linear:', r2*100,'%')
# plt.figure(figsize=(10, 6))
# plt.scatter(y_test, prediction, color='blue', label='Actual')
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red', label='Predicted')
# plt.title('Linear Regression: Actual vs Predicted')
# plt.xlabel('Actual Price')
# plt.ylabel('Predicted Price')
# plt.legend()
# plt.show()


#Polynomial Regression
poly_features = PolynomialFeatures(degree=2)
# transforms the existing features to higher degree features.
X_train_poly = poly_features.fit_transform(X_train)
# fit the transformed features to Linear Regression
poly_model = linear_model.LinearRegression()
poly_model.fit(X_train_poly, y_train)
# predicting on training data-set
y_train_predicted = poly_model.predict(X_train_poly)
ypred=poly_model.predict(poly_features.transform(X_test))
# predicting on test data-set
prediction = poly_model.predict(poly_features.fit_transform(X_test))
# Mean Squared Error
print('Mean Square Error Polynomial', metrics.mean_squared_error(y_test, prediction))
features = X_no_outliers.columns
target = Y_no_outliers.name
# Calculate R-squared score
r2 = r2_score(y_test, prediction)
print('R2 score polynomial:', r2*100,'%')
# plt.figure(figsize=(10, 6))
# plt.scatter(y_test, prediction, color='blue', label='Actual')
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red', label='Predicted')
# plt.title('Polynomial Regression: Actual vs Predicted')
# plt.xlabel('Actual Price')
# plt.ylabel('Predicted Price')
# plt.legend()
# plt.show()


#Random Forest Regression
rf_model = RandomForestRegressor(n_estimators=150, random_state=10)
# Fit the model to the training data
rf_model.fit(X_train_poly, y_train)
# Predict on the test set
rf_prediction = rf_model.predict(poly_features.transform(X_test))
# Calculate Mean Squared Error for Test Set
rf_test_mse = mean_squared_error(y_test, rf_prediction)
print("Random Forest Mean Squared Error (MSE) - Test:", rf_test_mse)
# Calculate R-squared score
rf_r2 = r2_score(y_test, rf_prediction)
print('Random Forest R2 score:', rf_r2*100,'%')
# plt.figure(figsize=(10, 6))
# plt.scatter(y_test, rf_prediction, color='blue', label='Actual')
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red', label='Predicted')
# plt.title('Random Forest Regression: Actual vs Predicted')
# plt.xlabel('Actual Price')
# plt.ylabel('Predicted Price')
# plt.legend()
# plt.show()


# Initialize SVR model
svr_model = SVR(kernel='poly',gamma=1)
# Fit the SVR model to the training data
svr_model.fit(X_train_poly, y_train)
# Predict on the test set
svr_prediction = svr_model.predict(poly_features.transform(X_test))
# Calculate Mean Squared Error for Test Set
svr_test_mse = mean_squared_error(y_test, svr_prediction)
print("SVR Mean Squared Error (MSE) - Test:", svr_test_mse)
# Calculate R-squared score
svr_r2 = r2_score(y_test, svr_prediction)
print('SVR R2 score:', svr_r2*100,'%')


# Initialize Gradient Boosting Regressor with specified hyperparameters
gb_model = GradientBoostingRegressor(n_estimators=100,
                                     learning_rate=0.3,
                                     random_state=10)
# Fit the Gradient Boosting model to the training data
gb_model.fit(X_train_poly, y_train)
# Predict on the test set
gb_prediction = gb_model.predict(poly_features.transform(X_test))
# Calculate Mean Squared Error for Test Set
gb_test_mse = mean_squared_error(y_test, gb_prediction)
print("Gradient Boosting Mean Squared Error (MSE) - Test:", gb_test_mse)
# Calculate R-squared score
gb_r2 = r2_score(y_test, gb_prediction)
print('Gradient Boosting R2 score:', gb_r2*100,'%')
# plt.figure(figsize=(10, 6))
# plt.scatter(y_test, gb_prediction, color='blue', label='Actual')
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red', label='Predicted')
# plt.title('Gradient Boosting Regression: Actual vs Predicted')
# plt.xlabel('Actual Price')
# plt.ylabel('Predicted Price')
# plt.legend()
# plt.show()


# Save the linear regression model
with open('linear_reg.pkl', 'wb') as linear_file:
    pickle.dump(model, linear_file)

# Save the polynomial regression model
with open('poly_features.pkl', 'wb') as poly_f_file:
    pickle.dump(poly_features, poly_f_file)

# Save the polynomial regression model
with open('poly_reg.pkl', 'wb') as poly_file:
    pickle.dump(poly_model, poly_file)

# Save the Random Forest Regression model
with open('random_forest.pkl', 'wb') as random_file:
    pickle.dump(rf_model, random_file)

# Save the SVR model
with open('svr.pkl', 'wb') as svr_file:
    pickle.dump(svr_model, svr_file)

# Save the Gradient Boosting Regression
with open('gradient_boosting.pkl', 'wb') as gb_file:
    pickle.dump(gb_model, gb_file)

#SAVING DATA
# Save mode_values and mean_values using pickle
with open('mode_values.pkl', 'wb') as f:
    pickle.dump(mode_values, f)

with open('mean_values.pkl', 'wb') as f:
    pickle.dump(mean_values, f)

with open('mode_rent.pkl', 'wb') as f:
    pickle.dump(mode_rent, f)

with open('mode_apartment.pkl', 'wb') as f:
    pickle.dump(mode_apartment, f)

with open('mode_housing.pkl', 'wb') as f:
    pickle.dump(mode_housing, f)
