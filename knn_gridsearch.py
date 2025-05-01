from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

def train_knn(X_train, Y_train):
    parameters = {
        'n_neighbors': list(range(1, 11)),
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'p': [1, 2]
    }
    knn = KNeighborsClassifier()
    knn_cv = GridSearchCV(knn, parameters, cv=10)
    knn_cv.fit(X_train, Y_train)
    print("Tuned hyperparameters (best parameters):", knn_cv.best_params_)
    print("Accuracy:", knn_cv.best_score_)
    return knn_cv
