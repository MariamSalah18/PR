# Apartment Rent Prediction

## Regression Analysis

### Data Preprocessing
- **Handling Missing Values:**
  - Filled missing values in 'bathrooms' and 'bedrooms' columns with mode.
  - Calculated mode values for 'amenities' separately for each category ('rent', 'apartment', 'housing') and filled missing values based on the mode of the respective category.
  - Filled missing values in 'pets allowed' column with 'None'.
  - Missing values in 'city name' and 'state' columns were filled with mode.
  - Addresses were filled based on the mode addresses for each city, with 'unknown' used as default if mode address isn't available.
  - Missing values in 'longitude' and 'latitude' columns were filled with the mean, and negative values in 'longitude' were made positive.

- **Handling Categorical Data:**
  - Applied label encoding to categorical columns using `LabelEncoder()` from `sklearn.preprocessing`.

- **Feature Scaling:**
  - Normalized features with small value ranges and standardized features with wide value ranges.

- **Feature Selection:**
  - Selected features based on correlation with the target column "price display".

- **Data Splitting:**
  - Split the data into training and testing sets with a test size of 30%.

### Model Selection
- Trained and evaluated linear regression, polynomial regression, support vector regression, random forest, and gradient boost models.

### Results
- Achieved varying accuracy scores and mean squared errors (MSE) for different models.
- Polynomial regression exhibited the lowest MSE and highest R2 score, indicating better performance compared to other models.

## Classification Analysis

### Changes in Preprocessing
- Removed standardization and used different methods for scaling data.
- Employed imputation with mean and mode to handle missing values.

### Feature Selection Process
- Utilized ANOVA, Kendall's Tau correlation, Chi-squared test, and Mutual Information to select informative features.

### Hyperparameter Tuning
- Tuned parameters for decision tree classifier, gradient boosting classifier, and SVM classifier.

### Conclusion
- Emphasized the importance of preprocessing steps in ensuring data quality and consistency.
- Highlighted the significance of feature selection strategies in streamlining model input and improving efficiency.
- Evaluated multiple classifiers and demonstrated the benefits of ensemble techniques.
- Advocated for continuous improvement through hyperparameter tuning and model optimization to enhance predictive performance.
