from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

def train_decision_tree(X_train, Y_train):
    parameters = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [2 * n for n in range(1, 10)],
        'max_features': ['sqrt'],  # 'auto' is deprecated for DecisionTreeClassifier
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10]
    }
    tree = DecisionTreeClassifier()
    tree_cv = GridSearchCV(tree, parameters, cv=10)
    tree_cv.fit(X_train, Y_train)
    print("Tuned hyperparameters (best parameters):", tree_cv.best_params_)
    print("Accuracy:", tree_cv.best_score_)
    return tree_cv
