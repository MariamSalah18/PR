import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from datetime import datetime
from sklearn.svm import SVC
from scipy.stats import kendalltau
from sklearn.feature_selection import chi2, mutual_info_classif, SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('ApartmentRentPrediction_Milestone2.csv')
X = df.drop(columns=['RentCategory'])
Y = pd.DataFrame(df['RentCategory'], columns=['RentCategory'])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
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
    # Calculate mode values for 'amenities' for each category
    mode_rent = X[X['category'] == 'rent']['amenities'].mode()[0]
    mode_apartment = X[X['category'] == 'apartment']['amenities'].mode()[0]
    mode_housing = X[X['category'] == 'housing']['amenities'].mode()[0]

    # Fill NaN values in 'amenities' based on category
    X.loc[X['category'] == 'rent', 'amenities'] = X.loc[X['category'] == 'rent', 'amenities'].fillna(mode_rent)
    X.loc[X['category'] == 'apartment', 'amenities'] = X.loc[X['category'] == 'apartment', 'amenities'].fillna(
        mode_apartment)
    X.loc[X['category'] == 'housing', 'amenities'] = X.loc[X['category'] == 'housing', 'amenities'].fillna(mode_housing)

    X['pets_allowed'].fillna('None', inplace=True)

    X['cityname'].fillna(X['cityname'].mode()[0], inplace=True)

    # Fill missing values in 'state' with mode
    X['state'].fillna(X['state'].mode()[0], inplace=True)

    X['address'] = X.groupby('cityname')['address'].transform(fill_mode)

    X['longitude'].fillna(X['longitude'].mean(), inplace=True)
    X['latitude'].fillna(X['latitude'].mean(), inplace=True)

    # Identify columns with object data type
    categorical_columns = X.select_dtypes(include=['object']).columns
    # Initialize LabelEncoder
    label_encoder = LabelEncoder()

    # Apply label encoding to each categorical column
    for column in categorical_columns:
        X[column] = label_encoder.fit_transform(X[column])

    # Replace outliers
    X = replace_outliers_with_mean(X)

    scaler = MinMaxScaler()
    for column in X:
        column_values = X[column].values.reshape(-1, 1)
        normalized_values = scaler.fit_transform(column_values)
        X[column] = normalized_values
    return X

X_train=preprocessing(X_train)
X_test=preprocessing(X_test)


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
#
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


# 1. Decision Tree Classifier
# Decision Tree Classifier with different hyperparameters
dt_classifier_1 = DecisionTreeClassifier(max_depth=3,criterion='gini')
dt_classifier_2 = DecisionTreeClassifier(max_depth=5,criterion='entropy')
dt_classifier_3 = DecisionTreeClassifier(max_depth=7,criterion='gini')

# Train each Decision Tree classifier
dt_classifier_1.fit(X_train, y_train)
dt_classifier_2.fit(X_train, y_train)
dt_classifier_3.fit(X_train, y_train)

# Predictions for each classifier
dt_predictions_1 = dt_classifier_1.predict(X_test)
dt_predictions_2 = dt_classifier_2.predict(X_test)
dt_predictions_3 = dt_classifier_3.predict(X_test)

# Calculate accuracy for each classifier
dt_accuracy_1 = accuracy_score(y_test, dt_predictions_1)
dt_accuracy_2 = accuracy_score(y_test, dt_predictions_2)
dt_accuracy_3 = accuracy_score(y_test, dt_predictions_3)

# Print accuracies
print("Decision Tree Classifier 1 Accuracy:", dt_accuracy_1 * 100)
print("Decision Tree Classifier 2 Accuracy:", dt_accuracy_2 * 100)
print("Decision Tree Classifier 3 Accuracy:", dt_accuracy_3 * 100)
print("---------------------------------------------------------------------------------------")

 # 2. Gradient Boosting Classifier
# Gradient Boosting (GBM) Classifier with different hyperparameters
gbm_classifier_1 = GradientBoostingClassifier(n_estimators= 150, learning_rate= 0.1)
gbm_classifier_2 = GradientBoostingClassifier(n_estimators= 250, learning_rate= 0.02)
gbm_classifier_3 = GradientBoostingClassifier(n_estimators= 300, learning_rate= 0.03)

# Train each GBM classifier
gbm_classifier_1.fit(X_train, y_train)
gbm_classifier_2.fit(X_train, y_train)
gbm_classifier_3.fit(X_train, y_train)

# Predictions for each classifier
gbm_predictions_1 = gbm_classifier_1.predict(X_test)
gbm_predictions_2 = gbm_classifier_2.predict(X_test)
gbm_predictions_3 = gbm_classifier_3.predict(X_test)

# Calculate accuracy for each classifier
gbm_accuracy_1 = accuracy_score(y_test, gbm_predictions_1)
gbm_accuracy_2 = accuracy_score(y_test, gbm_predictions_2)
gbm_accuracy_3 = accuracy_score(y_test, gbm_predictions_3)

# Print accuracies
print("Gradient Boosting Classifier 1 Accuracy:", gbm_accuracy_1 * 100)
print("Gradient Boosting Classifier 2 Accuracy:", gbm_accuracy_2 * 100)
print("Gradient Boosting Classifier 3 Accuracy:", gbm_accuracy_3 * 100)
print("---------------------------------------------------------------------------------------")

# 3. Support Vector Machine (SVM) Classifier
# Support Vector Machine (SVM) Classifier with different hyperparameters
svm_classifier_1 = SVC(kernel='linear', C=1)
svm_classifier_2 = SVC(kernel='rbf', gamma=0.8, C=0.5)
svm_classifier_3 = SVC(kernel='poly', degree=4, C=1)


# Train each SVM classifier
svm_classifier_1.fit(X_train, y_train)
svm_classifier_2.fit(X_train, y_train)
svm_classifier_3.fit(X_train, y_train)

# Predictions for each classifier
svm_predictions_1 = svm_classifier_1.predict(X_test)
svm_predictions_2 = svm_classifier_2.predict(X_test)
svm_predictions_3 = svm_classifier_3.predict(X_test)
# Calculate accuracy for each classifier
svm_accuracy_1 = accuracy_score(y_test, svm_predictions_1)
svm_accuracy_2 = accuracy_score(y_test, svm_predictions_2)
svm_accuracy_3 = accuracy_score(y_test, svm_predictions_3)

# Print accuracies
print("Support Vector Machine (SVM) Classifier 1 Accuracy:", svm_accuracy_1 * 100)
print("Support Vector Machine (SVM) Classifier 2 Accuracy:", svm_accuracy_2 * 100)
print("Support Vector Machine (SVM) Classifier 3 Accuracy:", svm_accuracy_3 * 100)
print("---------------------------------------------------------------------------------------")

# Define base models
base_models = [
    ('dt_1', DecisionTreeClassifier(max_depth=3, criterion='gini')),
    ('dt_2', DecisionTreeClassifier(max_depth=5, criterion='entropy')),
    ('dt_3', DecisionTreeClassifier(max_depth=7, criterion='gini')),
    ('gbm_1', GradientBoostingClassifier(n_estimators=150, learning_rate=0.1)),
    ('gbm_2', GradientBoostingClassifier(n_estimators=250, learning_rate=0.02)),
    ('gbm_3', GradientBoostingClassifier(n_estimators=300, learning_rate=0.03)),
    ('svm_1', SVC(kernel='linear', C=1)),
    ('svm_2', SVC(kernel='rbf', gamma=0.8, C=0.5)),
    ('svm_3', SVC(kernel='poly', degree=4, C=1))
]

# Initialize weighted voting ensemble
weighted_voting_ensemble = VotingClassifier(estimators=base_models, voting='hard')

# Train weighted voting ensemble
weighted_voting_ensemble.fit(X_train, y_train)

# Predictions from weighted voting ensemble
weighted_voting_predictions = weighted_voting_ensemble.predict(X_test)

# Calculate accuracy for weighted voting ensemble
weighted_voting_accuracy = accuracy_score(y_test, weighted_voting_predictions)
print("Weighted Voting Ensemble Accuracy:", weighted_voting_accuracy * 100)

# Initialize weighted stacking ensemble
stacking_ensemble = StackingClassifier(estimators=base_models)

# Train stacking ensemble
stacking_ensemble.fit(X_train, y_train)

# Predictions from stacking ensemble
stacking_predictions = stacking_ensemble.predict(X_test)

# Calculate accuracy for stacking ensemble
stacking_accuracy = accuracy_score(y_test, stacking_predictions)
print("Stacking Ensemble Accuracy:", stacking_accuracy * 100)

# SVMmax=max(svm_accuracy_1,svm_accuracy_2,svm_accuracy_3)
# gbnmax=max(gbm_accuracy_1,gbm_accuracy_2,gbm_accuracy_3)
# DTmax=max(dt_accuracy_1,dt_accuracy_2,dt_accuracy_3)
#
# accuracies = [DTmax, gbnmax, SVMmax]
#
# # Voting Approach
# voting_clf = VotingClassifier(
#     estimators=[
#         ('dt_1', dt_classifier_1),
#         ('dt_2', dt_classifier_2),
#         ('dt_3', dt_classifier_3),
#         ('gbm_1', gbm_classifier_1),
#         ('gbm_2', gbm_classifier_2),
#         ('gbm_3', gbm_classifier_3),
#         ('svm_1', svm_classifier_1),
#         ('svm_2', svm_classifier_2),
#         ('svm_3', svm_classifier_3)
#     ],
#     voting='hard'  # Use majority voting
# )
#
# voting_clf.fit(X_train, y_train)
# voting_predictions = voting_clf.predict(X_test)
# voting_accuracy = accuracy_score(y_test, voting_predictions)
# print("Voting Classifier Accuracy:", voting_accuracy * 100)
#
# # Stacking Approach
# estimators = [
#     ('dt_1', dt_classifier_1),
#     ('dt_2', dt_classifier_2),
#     ('dt_3', dt_classifier_3),
#     ('gbm_1', gbm_classifier_1),
#     ('gbm_2', gbm_classifier_2),
#     ('gbm_3', gbm_classifier_3),
#     ('svm_1', svm_classifier_1),
#     ('svm_2', svm_classifier_2),
#     ('svm_3', svm_classifier_3)
# ]
#
# stacking_clf = StackingClassifier(
#     estimators=estimators,
#     final_estimator=LogisticRegression()  # Meta-classifier
# )
#
# stacking_clf.fit(X_train, y_train)
# stacking_predictions = stacking_clf.predict(X_test)
# stacking_accuracy = accuracy_score(y_test, stacking_predictions)
# print("Stacking Classifier Accuracy:", stacking_accuracy * 100)



# SVMmax=max(svm_accuracy_1,svm_accuracy_2,svm_accuracy_3)
# gbnmax=max(gbm_accuracy_1,gbm_accuracy_2,gbm_accuracy_3)
# DTmax=max(dt_accuracy_1,dt_accuracy_2,dt_accuracy_3)
#
# # Plotting
# classifiers = ['Decision Tree', 'Gradient Boosting', 'SVM']
# accuracies = [DTmax, gbnmax, SVMmax]
#
# plt.bar(classifiers, accuracies, color='skyblue')
#
# # Highlighting maximum accuracy for each classifier type
# for i, acc in enumerate(accuracies):
#     plt.text(i, acc, f'{acc:.2f}', ha='center', va='bottom')
#
# plt.title('Max Accuracy of Each Classifier Type')
# plt.xlabel('Classifier')
# plt.ylabel('Accuracy')
# plt.ylim(0, 1)
# plt.show()
