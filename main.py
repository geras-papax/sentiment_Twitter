from dataProcess import getData
from naiveBayes import naive_bayes
from logisticRegression import logistic_regression
from lstm import lstm, preprocess, prediction

X_train, X_test, y_train, y_test, raw_X_train, raw_X_test = getData()
naive_bayes(X_train, X_test, y_train, y_test)
logistic_regression(X_train, X_test, y_train, y_test)
tokenizer,model = lstm(raw_X_train, raw_X_test, y_train,  y_test)
print("Insert your Tweet:")
print("------------------")
text = input()
print("------------------")
prediction(model, tokenizer, text)

