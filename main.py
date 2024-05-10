import numpy as np
import pandas as pd
import warnings
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from datetime import datetime, time
import time
from sklearn.svm import SVC
from scipy.stats import kendalltau
from sklearn.feature_selection import chi2, mutual_info_classif, SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
import pickle
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
warnings.filterwarnings('ignore')

df = pd.read_csv('ApartmentRentPrediction_Milestone2.csv')
X = df.drop(columns=['RentCategory'])
Y = pd.DataFrame(df['RentCategory'], columns=['RentCategory'])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
label_encoder=0
scaler=0
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

#colunms_names=['bathrooms','bedrooms','category','amenities','cityname','state','longitude','latitude']
# Lists to store mean and mode
mean_values = []
mode_values = []
def preprocessing(X):
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
    X['category'] = X['category'].fillna(X['category'].mode()[0])
    X['amenities'] = X['amenities'].fillna(X['amenities'].mode()[0])
    X['pets_allowed'].fillna('None', inplace=True)
    X['cityname'].fillna(X['cityname'].mode()[0], inplace=True)
    X['state'].fillna(X['state'].mode()[0], inplace=True)
    X['address'] = X.groupby('cityname')['address'].transform(fill_mode)
    X['longitude'].fillna(X['longitude'].mean(), inplace=True)
    X['latitude'].fillna(X['latitude'].mean(), inplace=True)
    categorical_columns = X.select_dtypes(include=['object']).columns
    # Initialize LabelEncoder for each categorical column and save them individually
    for column in categorical_columns:
        label_encoder = LabelEncoder()
        X[column] = label_encoder.fit_transform(X[column])
        # Save the LabelEncoder object
        with open(f"label_encoder_{column}.pkl", "wb") as file:
            pickle.dump(label_encoder, file)
    for col in  X.columns:
            # Fill missing values with the mean and save mean
            mean_val = X[col].mean()
            mean_values.append(mean_val)
            X[col].fillna(mean_val, inplace=True)
            # Fill missing values with the mode and save mode
            mode_val = X[col].mode()[0]
            mode_values.append(mode_val)
            X[col] = X[col].fillna(mode_val)
    # replace outliers
    X = replace_outliers_with_mean(X)
    # scaler = MinMaxScaler()
    for column in X:
        scaler = MinMaxScaler()
        column_values = X[column].values.reshape(-1, 1)
        normalized_values = scaler.fit_transform(column_values)
        X[column] = normalized_values
        with open(f"normalized_values_{column}.pkl", "wb") as file:
            pickle.dump(scaler, file)
    return X

X_train=preprocessing(X_train)
X_test=preprocessing(X_test)
#
# print(mean_values)
# print(mode_values)

# print("Number of values for each column in label_encoders_train dictionary:")
# for column, encoder in label_encoders_train.items():
#     if isinstance(encoder, LabelEncoder):
#         print(f"{column}: {len(encoder.classes_)}")
#
# Define the subset of features "numeric feature"
selected_features = ['id', 'bathrooms', 'bedrooms', 'square_feet', 'latitude', 'longitude', 'time']

# Perform feature selection using ANOVA
anova_scores, p_values_anova = f_classif(X_train[selected_features], y_train)

# Initialize lists to store Kendall scores and p-values
kendall_scores = []
p_values_kendall = []

# Compute Kendall's tau correlation coefficient and p-value for each feature
for column in selected_features:
    kendall_score, p_value = kendalltau(X_train[column], y_train)
    kendall_scores.append(kendall_score)
    p_values_kendall.append(p_value)

# Create DataFrames to store feature scores and p-values
anova_results = pd.DataFrame({'Feature': selected_features, 'ANOVA Score': anova_scores, 'p-value ANOVA': p_values_anova})
kendall_results = pd.DataFrame({'Feature': selected_features, 'Kendall Score': kendall_scores, 'p-value Kendall': p_values_kendall})

# Filter significant features based on a threshold (e.g., p-value < 0.05)
significant_anova_features = anova_results[anova_results['p-value ANOVA'] < 0.05]
significant_kendall_features = kendall_results[kendall_results['p-value Kendall'] < 0.05]
#
# # Print significant features
# print("Significant Features Selected by ANOVA:")
# print(significant_anova_features)
# print("\nSignificant Features Selected by Kendall's Tau Correlation:")
# print(significant_kendall_features)
# Define the subset of features
selected_features = ['category', 'title', 'body', 'amenities', 'currency', 'fee',
                     'has_photo','pets_allowed','price_type','address','state','source','cityname']

# Apply Chi-Squared test
chi2_scores, chi2_p_values = chi2(X_train[selected_features], y_train)

# Apply Mutual Information
y_train = np.ravel(y_train)
mutual_info_scores = mutual_info_classif(X_train[selected_features], y_train)

# Select the top features based on Chi-Squared and Mutual Information
k_best_chi2 = SelectKBest(chi2, k=3)  # Select top 3 features based on Chi-Squared
X_chi2_selected = k_best_chi2.fit_transform(X_train[selected_features], y_train)

k_best_mi = SelectKBest(mutual_info_classif, k=3)  # Select top 3 features based on Mutual Information
X_mi_selected = k_best_mi.fit_transform(X_train[selected_features], y_train)

# Get the selected feature indices
selected_chi2_indices = k_best_chi2.get_support(indices=True)
selected_mi_indices = k_best_mi.get_support(indices=True)

# Get the names of the selected features
selected_chi2_features = [selected_features[i] for i in selected_chi2_indices]
selected_mi_features = [selected_features[i] for i in selected_mi_indices]

# print("Top 3 features selected by Chi-Squared:")
# print(selected_chi2_features)
# print("\nTop 3 features selected by Mutual Information:")
# print(selected_mi_features)

# unique_counts = X_train.nunique().sort_values()
# print(unique_counts)

selected_features = ['bathrooms', 'bedrooms', 'square_feet', 'latitude', 'longitude',
                     'time', 'has_photo', 'address', 'state', 'cityname']
X_train=X_train[selected_features]
X_test=X_test[selected_features]

# Train and save Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(max_depth=7, criterion='gini')
start_time_dt_classifier = time.time()
dt_classifier.fit(X_train, y_train)
end_train_time_dt_classifier = time.time()
training_time_dt_classifier = end_train_time_dt_classifier - start_time_dt_classifier
start_pred_time_dt_classifier = time.time()
dt_predictions = dt_classifier.predict(X_test)
end_test_time_dt_classifier = time.time()
test_time_dt_classifier = end_test_time_dt_classifier - start_pred_time_dt_classifier
dt_accuracy = accuracy_score(y_test, dt_predictions)
print("Decision Tree Classifier Accuracy:", dt_accuracy * 100)
# print("Training Time:", training_time_dt_classifier, "seconds")
# print("Prediction Time:", test_time_dt_classifier, "seconds")

# Train and save Gradient Boosting Classifier
gbm_classifier = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1)
start_time_gbm_classifier = time.time()
gbm_classifier.fit(X_train, y_train)
end_train_time_gbm_classifier = time.time()
training_time_gbm_classifier = end_train_time_gbm_classifier - start_time_gbm_classifier
start_pred_time_gbm_classifier  = time.time()
gbm_predictions = gbm_classifier.predict(X_test)
end_test_time_gbm_classifier  = time.time()
test_time_gbm_classifier  = end_test_time_gbm_classifier  - start_pred_time_gbm_classifier
gbm_accuracy = accuracy_score(y_test, gbm_predictions)
print("Gradient Boosting Classifier Accuracy:", gbm_accuracy * 100)

# Train and save Support Vector Machine Classifier
svm_classifier = SVC(kernel='rbf', gamma=0.8, C=0.5)
start_time_svm_classifier = time.time()
svm_classifier.fit(X_train, y_train)
end_train_time_svm_classifier = time.time()
training_time_svm_classifier = end_train_time_svm_classifier - start_time_svm_classifier
start_pred_time_svm_classifier  = time.time()
svm_predictions = svm_classifier.predict(X_test)
end_test_time_svm_classifier = time.time()
test_time_svm_classifier  = end_test_time_svm_classifier  - start_pred_time_svm_classifier
svm_accuracy = accuracy_score(y_test, svm_predictions)
print("Support Vector Machine (SVM) Classifier Accuracy:", svm_accuracy * 100)

# Define base models
base_models = [
    ('dt', dt_classifier),
    ('gbm', gbm_classifier),
    ('svm', svm_classifier)
]

# Calculate total accuracy
total_accuracy = svm_accuracy + gbm_accuracy + dt_accuracy

# Compute weights based on normalized accuracies
weights = {
    'svm': svm_accuracy / total_accuracy,
    'gbm': gbm_accuracy / total_accuracy,
    'dt': dt_accuracy / total_accuracy,
}

# Initialize weighted voting ensemble with weights
weighted_voting_ensemble = VotingClassifier(estimators=base_models, voting='hard', weights=list(weights.values()))

# Train weighted voting ensemble
weighted_voting_ensemble.fit(X_train, y_train)


# Predictions from weighted voting ensemble
weighted_voting_predictions = weighted_voting_ensemble.predict(X_test)

# Calculate accuracy for weighted voting ensemble
weighted_voting_accuracy = accuracy_score(y_test, weighted_voting_predictions)
print("Weighted Voting Ensemble Accuracy:", weighted_voting_accuracy * 100)

# Initialize weighted stacking ensemble with weights
stacking_ensemble = StackingClassifier(estimators=base_models,
                                       stack_method='auto',
                                       passthrough=True,
                                       final_estimator=KNeighborsClassifier())

# Train stacking ensemble
stacking_ensemble.fit(X_train, y_train)


# Predictions from stacking ensemble
stacking_predictions = stacking_ensemble.predict(X_test)

# Calculate accuracy for stacking ensemble
stacking_accuracy = accuracy_score(y_test, stacking_predictions)
print("Stacking Ensemble Accuracy:", stacking_accuracy * 100)

# Define base models
base_models = [
    ('dt', dt_classifier),
    ('gbm', gbm_classifier)]


# Calculate total accuracy
total_accuracy = gbm_accuracy + dt_accuracy

# Compute weights based on normalized accuracies
weights = {
    'gbm': gbm_accuracy / total_accuracy,
    'dt': dt_accuracy / total_accuracy,
}

# Initialize weighted voting ensemble with weights
weighted_voting_ensemble = VotingClassifier(estimators=base_models, voting='hard', weights=list(weights.values()))

# Train weighted voting ensemble
weighted_voting_ensemble.fit(X_train, y_train)


# Predictions from weighted voting ensemble
weighted_voting_predictions = weighted_voting_ensemble.predict(X_test)


# Calculate accuracy for weighted voting ensemble
weighted_voting_accuracy = accuracy_score(y_test, weighted_voting_predictions)
print("Weighted Voting Ensemble Accuracy :", weighted_voting_accuracy * 100)

# Initialize weighted stacking ensemble with weights
stacking_ensemble = StackingClassifier(estimators=base_models,
                                       stack_method='auto',
                                       passthrough=True,
                                       final_estimator=KNeighborsClassifier())

# Train stacking ensemble
stacking_ensemble.fit(X_train, y_train)

# Predictions from stacking ensemble
stacking_predictions = stacking_ensemble.predict(X_test)

# Calculate accuracy for stacking ensemble
stacking_accuracy = accuracy_score(y_test, stacking_predictions)
print("Stacking Ensemble Accuracy:", stacking_accuracy * 100)

#SAVING THE MODEL
# Save Decision Tree Classifier
#file name,write binary
file_name='dt_classifier.pkl'
pickle.dump(dt_classifier, open(file_name,'wb'))

# Save Gradient Boosting Classifier
#file name,write binary
file_name='gbm_classifier.pkl'
pickle.dump(gbm_classifier, open(file_name,'wb'))

 # Save Support Vector Machine Classifier
#file name,write binary
file_name='svm_classifier.pkl'
pickle.dump(gbm_classifier, open(file_name,'wb'))

# Save weighted voting ensemble
file_name='weighted_voting_ensemble.pkl'
pickle.dump(weighted_voting_ensemble,open(file_name,'wb'))

# Save stacking ensemble
file_name='stacking_ensemble.pkl'
pickle.dump(stacking_ensemble,open(file_name,'wb'))

# Save weighted voting ensemble
file_name='weighted_voting_ensemble_2.pkl'
pickle.dump(weighted_voting_ensemble,open(file_name,'wb'))

# Save stacking ensemble
file_name='stacking_ensemble_2.pkl'
pickle.dump(stacking_ensemble,open(file_name,'wb'))


# Save mode_values and mean_values using pickle
with open('mode_values.pkl', 'wb') as f:
    pickle.dump(mode_values, f)

with open('mean_values.pkl', 'wb') as f:
    pickle.dump(mean_values, f)

# Plotting
classifiers = ['Decision Tree', 'Gradient Boosting', 'SVM']
accuracies = [dt_accuracy*100, gbm_accuracy*100, svm_accuracy*100]

plt.bar(classifiers, accuracies, color='skyblue')

# Highlighting maximum accuracy for each classifier type
for i, acc in enumerate(accuracies):
    plt.text(i, acc, f'{acc:.2f}', ha='center', va='bottom')

plt.title('Max Accuracy of Each Classifier Type')
plt.xlabel('Classifier')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.show()

# Sample data
class_accuracy = {'DT': dt_accuracy, 'GBM': gbm_accuracy, 'SVM': svm_accuracy}
training_time = {'DT': training_time_dt_classifier, 'GBM': training_time_gbm_classifier, 'SVM': training_time_svm_classifier} # in seconds
test_time = {'DT': test_time_dt_classifier, 'GBM': test_time_gbm_classifier, 'SVM': test_time_svm_classifier} # in seconds

# Bar graph for total training time
plt.figure(figsize=(8, 5))
plt.bar(training_time.keys(), training_time.values(), color='lightgreen')
plt.title('Total Training Time')
plt.xlabel('Model')
plt.ylabel('Time (seconds)')
plt.show()

# Bar graph for total test time
plt.figure(figsize=(8, 5))
plt.bar(test_time.keys(), test_time.values(), color='salmon')
plt.title('Total Test Time')
plt.xlabel('Model')
plt.ylabel('Time (seconds)')
plt.show()