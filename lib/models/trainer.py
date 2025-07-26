from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_and_evaluate(X_train, y_train, X_test, y_test, model_type='logistic'):
    if model_type == 'naive_bayes':
        model = MultinomialNB()
    elif model_type == 'logistic':
        model = LogisticRegression(penalty='l2', max_iter=1000)  # L2 regularization
    else:
        raise ValueError("model_type must be either 'logistic' or 'naive_bayes'")

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)

    return model, predictions, acc
