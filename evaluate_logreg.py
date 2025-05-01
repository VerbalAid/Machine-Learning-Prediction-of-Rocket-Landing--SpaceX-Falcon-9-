from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np

# Define parameter grid
parameters = {
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'C': np.logspace(-3, 3, 5),
    'gamma': np.logspace(-3, 3, 5)
}

# Create SVM object
svm = SVC()

# Create and fit GridSearchCV
svm_cv = GridSearchCV(svm, parameters, cv=10)
svm_cv.fit(X_train, Y_train)

# Output best parameters and score
print("Tuned hyperparameters: ", svm_cv.best_params_)
print("Accuracy:", svm_cv.best_score_)
