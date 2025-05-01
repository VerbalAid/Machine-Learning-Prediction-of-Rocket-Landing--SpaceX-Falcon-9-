from data_loader import load_data
from data_preprocessing import extract_target_variable, standardize_features
from model_training import split_data, train_logistic_regression

# Load and prepare data
data, X = load_data()  
Y = extract_target_variable(data)
X = standardize_features(X)

# Split
X_train, X_test, Y_train, Y_test = split_data(X, Y)

# Train model
logreg_cv = train_logistic_regression(X_train, Y_train)

# Output results
print("Tuned hyperparameters:", logreg_cv.best_params_)
print("Best cross-validated accuracy:", logreg_cv.best_score_)
