#http://stackoverflow.com/questions/30056331/how-to-list-all-scikit-learn-classifiers-that-support-predict-proba
from sklearn.utils.testing import all_estimators

estimators = all_estimators()

for name, class_ in estimators:
    if hasattr(class_, 'predict_proba'):
        print(name)
