
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import seaborn as sns
from itertools import combinations
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

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
