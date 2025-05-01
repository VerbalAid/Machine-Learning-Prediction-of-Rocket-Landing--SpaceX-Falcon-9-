from sklearn.metrics import accuracy_score
from confusion_matrix_plot import plot_confusion_matrix

def evaluate_svm(svm_model, X_test, Y_test):
    yhat = svm_model.predict(X_test)
    acc = accuracy_score(Y_test, yhat)
    plot_confusion_matrix(Y_test, yhat)
    return acc
