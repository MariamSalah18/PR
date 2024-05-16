import pickle
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')
default_value="unknown"
def fill_mode(x):
    if not x.mode().empty:
        return x.mode().iloc[0]  # Use mode if available
    else:
       return default_value


# Load saved models
#file name,read binary
dt_classifier_loaded=pickle.load(open('dt_classifier.pkl', 'rb'))
gbm_classifier_loaded=pickle.load(open('gbm_classifier.pkl', 'rb'))
svm_classifier_loaded=pickle.load(open('svm_classifier.pkl', 'rb'))
weighted_voting_ensemble_loaded = pickle.load(open('weighted_voting_ensemble.pkl', 'rb'))
stacking_ensemble_loaded = pickle.load(open('stacking_ensemble.pkl', 'rb'))
weighted_voting_ensemble_2_loaded = pickle.load(open('weighted_voting_ensemble_2.pkl', 'rb'))
stacking_ensemble_2_loaded = pickle.load(open('stacking_ensemble_2.pkl', 'rb'))
mean_values_loaded = pickle.load(open('mean_values.pkl', 'rb'))
mode_values_loaded = pickle.load(open('mode_values.pkl', 'rb'))

test_data=pd.read_csv("ApartmentRentPrediction_Milestone2.csv")

selected_features = ['bathrooms', 'bedrooms', 'square_feet', 'latitude', 'longitude',
                     'time', 'has_photo', 'address', 'state', 'cityname']

X = test_data.drop(columns=['RentCategory'])
Y = pd.DataFrame(test_data['RentCategory'], columns=['RentCategory'])

X_columns = X.columns
# Using dictionary comprehension to create the dictionary
mean_dict = {column_name: mean_value for column_name, mean_value in zip(X_columns, mean_values_loaded)}
mode_dict = {column_name: mode_value for column_name, mode_value in zip(X_columns, mode_values_loaded)}

X=X[selected_features]

X['time'] = [datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') for ts in X['time']]

X['bathrooms'] = X['bathrooms'].fillna(mode_dict['bathrooms']).astype(int)
X['bedrooms'].replace(0, 1, inplace=True)
X['bedrooms'] = X['bedrooms'].fillna(mode_dict['bedrooms'])
X['cityname'] = X['cityname'].fillna(mode_dict['cityname'])
X['state'] = X['state'].fillna(mode_dict['state'])
X['longitude'] = X['longitude'].fillna(mean_dict['longitude'])
X['latitude'] = X['latitude'].fillna(mean_dict['latitude'])

X['address'] = X.groupby('cityname')['address'].transform(fill_mode)
label_encoders = {}
scalers = {}
categorical_columns = ['has_photo', 'address', 'cityname', 'state', 'time']
# Load LabelEncoders and handle unseen labels
for column in categorical_columns:
    with open(f"label_encoder_{column}.pkl", "rb") as file:
        label_encoder = pickle.load(file)
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

# Make predictions using the loaded models on the test data
dt_predictions = dt_classifier_loaded.predict(X)
gbm_predictions = gbm_classifier_loaded.predict(X)
svm_predictions = svm_classifier_loaded.predict(X)
weighted_voting_predictions = weighted_voting_ensemble_loaded.predict(X)
stacking_predictions = stacking_ensemble_loaded.predict(X)
weighted_voting_predictions_2 = weighted_voting_ensemble_2_loaded.predict(X)
stacking_predictions_2 = stacking_ensemble_2_loaded.predict(X)

# Calculate accuracy for each model
dt_accuracy = accuracy_score(Y, dt_predictions)
gbm_accuracy = accuracy_score(Y, gbm_predictions)
svm_accuracy = accuracy_score(Y, svm_predictions)
weighted_voting_accuracy = accuracy_score(Y, weighted_voting_predictions)
stacking_accuracy = accuracy_score(Y, stacking_predictions)
weighted_voting_accuracy_2 = accuracy_score(Y, weighted_voting_predictions_2)
stacking_accuracy_2 = accuracy_score(Y, stacking_predictions_2)

# Print the accuracies
print("Decision Tree Accuracy:", dt_accuracy * 100)
print("Gradient Boosting Accuracy:", gbm_accuracy * 100)
print("SVM Accuracy:", svm_accuracy * 100)
print("Weighted Voting Accuracy:", weighted_voting_accuracy * 100)
print("Stacking Accuracy:", stacking_accuracy * 100)
print("Weighted Voting Accuracy 2:", weighted_voting_accuracy_2 * 100)
print("Stacking Accuracy 2:", stacking_accuracy_2 * 100)