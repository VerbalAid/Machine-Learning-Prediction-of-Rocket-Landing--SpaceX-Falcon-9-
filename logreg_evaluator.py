from sklearn.metrics import plot_confusion_matrix

# Accuracy on test data
print("Test Accuracy:", logreg_cv.score(X_test, Y_test))

# Confusion matrix
yhat = logreg_cv.predict(X_test)
plot_confusion_matrix(logreg_cv, X_test, Y_test)
