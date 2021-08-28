from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics

def fitness(Best,data,label):
    data_FS = data[:,Best]
    return svm_model(data_FS,label)

def svm_model(data, result):
    train_x, test_x, train_y, test_y = train_test_split(data, result, test_size=0.3)
    clf = SVC()
    try:
        clf.fit(train_x, train_y.flatten())
        predicted = clf.predict(test_x)
        return metrics.accuracy_score(test_y, predicted)
    except Exception:
        return 0