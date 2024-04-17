
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from itertools import combinations
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from scipy import stats

df = pd.read_csv('ApartmentRentPrediction.csv')
X = df.drop(columns=['price_display'])
Y= df['price_display']
def set_category(text):
    text_lower = text.lower()  # Convert text to lowercase
    if 'rent' in text_lower:
        return 'rent'
    elif 'apartment' in text_lower:
        return 'apartment'
    else:
        return 'housing'
X['category'] = X['body'].apply(set_category)

X['bathrooms'].replace([np.inf, -np.inf], 1, inplace=True)

# Fill NaN values in 'bathrooms' based on 'category'
X.loc[X['category'] == 'rent', 'bathrooms'] = X.loc[X['category'] == 'rent', 'bathrooms'].fillna(1)
X.loc[X['category'] == 'apartment', 'bathrooms'] = X.loc[X['category'] == 'apartment', 'bathrooms'].fillna(2)
X.loc[X['category'] == 'housing', 'bathrooms'] = X.loc[X['category'] == 'housing', 'bathrooms'].fillna(3)



X['bathrooms'].replace(0, 1, inplace=True)
# Fill NaN values in 'bathrooms' based on 'category'
X.loc[X['category'] == 'rent', 'bedrooms'] = X.loc[X['category'] == 'rent', 'bedrooms'].fillna(1)
X.loc[X['category'] == 'apartment', 'bedrooms'] = X.loc[X['category'] == 'apartment', 'bedrooms'].fillna(2)
X.loc[X['category'] == 'housing', 'bedrooms'] = X.loc[X['category'] == 'housing', 'bedrooms'].fillna(3)

'''X.loc[X['category'] == 'rent', 'amenities'].fillna(X['amenities'].mean(),inplace=True)
X.loc[X['category'] == 'apartment', 'amenities'].fillna(X['amenities'].mean(),inplace=True)
X.loc[X['category'] == 'housing', 'amenities'].fillna(X['amenities'].mean(),inplace=True)'''

#X['amenities'].fillna('None', inplace=True)

# Calculate mode values for 'amenities' for each category
mode_rent = X[X['category'] == 'rent']['amenities'].mode()[0]
mode_apartment = X[X['category'] == 'apartment']['amenities'].mode()[0]
mode_housing = X[X['category'] == 'housing']['amenities'].mode()[0]

# Fill NaN values in 'amenities' based on category
X.loc[X['category'] == 'rent', 'amenities']=X.loc[X['category'] == 'rent', 'amenities'].fillna(mode_rent)
X.loc[X['category'] == 'apartment', 'amenities']=X.loc[X['category'] == 'apartment', 'amenities'].fillna(mode_apartment)
X.loc[X['category'] == 'housing', 'amenities']=X.loc[X['category'] == 'housing', 'amenities'].fillna(mode_housing)

X['pets_allowed'].fillna('None', inplace=True)

X['cityname'].fillna(X['cityname'].mode()[0], inplace=True)

# Fill missing values in 'state' with mode
X['state'].fillna(X['state'].mode()[0], inplace=True)


# Find cities with only one address
cities_with_single_address = X.groupby('cityname')['address'].nunique()
cities_with_single_address = cities_with_single_address[cities_with_single_address == 1]

# Define a function to fill missing addresses with mode addresses if available, otherwise with random addresses

default_value="unknown"
def fill_mode(x):
    if not x.mode().empty:
        return x.mode().iloc[0]  # Use mode if available
    else:
       return default_value

# Apply the function to fill missing addresses based on mode addresses or single addresses for each city

X['address'] = X.groupby('cityname')['address'].transform(fill_mode)

X['longitude'] = X['longitude'].abs()
X['longitude'].fillna(X['longitude'].mean(), inplace=True)
X['latitude'].fillna(X['latitude'].mean(), inplace=True)

# Identify columns with object data type
categorical_columns = X.select_dtypes(include=['object']).columns
# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Apply label encoding to each categorical column
for column in categorical_columns:
    X[column] = label_encoder.fit_transform(X[column])


def remove_outliers(df):
    # Calculate z-scores for each value in the DataFrame
    z_scores = np.abs(stats.zscore(df))

    # Define a threshold to identify outliers
    threshold = 3

    # Create a boolean mask of outliers for each column
    outlier_mask = (z_scores > threshold).any(axis=1)

    # Remove rows containing outliers
    df_clean = df[~outlier_mask]

    return df_clean

X_no_outliers = remove_outliers(X)

scaler = MinMaxScaler()

columns_to_normalize = ['category', 'amenities', 'bathrooms','bedrooms'
    ,'has_photo','pets_allowed','price','square_feet','address','cityname','state','latitude','longitude','source']

for column in columns_to_normalize:
    column_values = X_no_outliers[column].values.reshape(-1, 1)
    normalized_values = scaler.fit_transform(column_values)
    X_no_outliers[column] = normalized_values

columns_to_standardize = ['title','body']
scaler = StandardScaler()
X_no_outliers[columns_to_standardize] = scaler.fit_transform(X_no_outliers[columns_to_standardize])





'''preprocessed_file_path = 'preprocessed_data.xlsx'
X_no_outliers.to_excel(preprocessed_file_path, index=False)
print("Preprocessed data exported to:", preprocessed_file_path)'''

'''sns.boxplot(X['price'])
plt.ylim(0, 50)
plt.show()
 preprocessed_file_path = 'preprocessed_data.xlsx'
X.to_excel(preprocessed_file_path, index=False)
print("Preprocessed data exported to:", preprocessed_file_path)

# Plot histograms for numerical columns
numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns"
#for column in numerical_columns:
 #   plt.figure(figsize=(8, 6))
  #  plt.hist(X[column], bins=20, color='skyblue', edgecolor='black')
   # plt.title(f'Histogram of {column}')
    #plt.xlabel(column)
    #plt.ylabel('Frequency')
    #plt.grid(True)
   # plt.show()'''

