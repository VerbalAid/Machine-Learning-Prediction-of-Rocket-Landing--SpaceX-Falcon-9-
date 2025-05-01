from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression

def split_data(X, Y, test_size=0.2, random_state=2):
    """
    Splits data into training and testing sets.
    """
    return train_test_split(X, Y, test_size=test_size, random_state=random_state)

def train_logistic_regression(X_train, Y_train):
    """
    Performs GridSearchCV to find the best logistic regression parameters.
    Returns the best model and its performance.
    """
    parameters = {
        'C': [0.01, 0.1, 1],
        'penalty': ['l2'],
        'solver': ['lbfgs']
    }
    lr = LogisticRegression()
    logreg_cv = GridSearchCV(lr, parameters, cv=10)
    logreg_cv.fit(X_train, Y_train)
    return logreg_cv
