import sklearn as sk
import numpy as np
import string
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.model_selection import cross_val_score
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from yellowbrick.model_selection import ValidationCurve

class vectorizer():
    def __init__(self):
        self.vectorizer = None

    def save(self, vectorizer, cv):
        self.vectorizer = vectorizer
        self.cv = cv

def fileIO(file_name):
    file = pd.read_csv(file_name, sep='\t', names=['Documents', 'Sentiment'])

    file.Documents = file.Documents.apply(lambda x: x.lower())
    file.Documents = file.Documents.apply(lambda x: x.translate(None, string.punctuation))
    file.Documents = file.Documents.apply(lambda x: x.translate(None, string.digits))

    return file

def bag_of_words(vectorizer, test):

    if test == False:
        cv = CV()
        data = fileIO('trainreviews.txt')
        labels = data.Sentiment
        data = cv.fit_transform(data.Documents)
        tfidf = sk.feature_extraction.text.TfidfTransformer()
        bag_o_words = tfidf.fit_transform(data)
        vectorizer.save(tfidf, cv)

        return bag_o_words, labels
    # otherwise test is true
    data = fileIO('testreviewsunlabeled.txt')
    data = vectorizer.cv.transform(data.Documents)
    bag_o_words = vectorizer.vectorizer.transform(data)
    return bag_o_words


def word_embeddings(test_file):
    stop = [str(word.decode('utf-8')) for word in stopwords.words('english')]
    glove = datapath('/Users/chase/Desktop/Tufts/Fall2018/COMP135/Project2/glove.6B.50d.txt')
    w2v_file = get_tmpfile("word2vec.txt")
    glove2word2vec(glove, w2v_file)
    w2v_model = KeyedVectors.load_word2vec_format(w2v_file)

    data = fileIO(test_file)
    data.Documents = data.Documents.apply(lambda x: [word for word in string.split(x) if word not in stop])

    new_features = []
    labels = []
    i = 0

    for sentence in data.Documents:
        w = np.zeros(50, dtype=float)

        for word in sentence:
            if word in w2v_model:
                w += w2v_model.get_vector(word)

        new_features.append(w)
        labels.append(data.Sentiment[i])
        i+=1

    X = np.array(new_features)
    Y = np.array(labels)

    return X, Y


def SVM(X_train,Y_train, X_test, Y_test):
    svm = sk.svm.LinearSVC()
    print(cross_val_score(svm, X_train, Y_train, cv= 10))
    svm = sk.svm().fit(X_train, Y_train)
    preds = svm.predict(X_test)
    accuracy = np.mean(preds == Y_test)
    return accuracy

def KNN(X_train,Y_train, X_test, Y_test, k):
    knn = sk.neighbors.KNeighborsClassifier(n_neighbors=k).fit(X_train, Y_train)
    preds = knn.predict(X_test)
    accuracy = np.mean(preds == Y_test)
    return accuracy

def LogReg(X_train,Y_train, X_test, Y_test):
    log_reg = sk.linear_model.LogisticRegression(solver='lbfgs').fit(X_train, Y_train)
    preds = log_reg.predict(X_test)
    accuracy = np.mean(preds == Y_test)
    return accuracy

def model_select(X,Y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.5, random_state=0)

    best_models = []
    svm_hp = [{'C': [.01, .1, 1, 10]}]
    svm = sk.svm.LinearSVC(max_iter=40000)

    knn_hp = [{'n_neighbors': [50, 75, 100, 125]}]
    knn = sk.neighbors.KNeighborsClassifier()

    log_reg_hp = [{'C': [100, 10, 1, 0.1]}]
    log_reg = sk.linear_model.LogisticRegression(solver='lbfgs', fit_intercept = True, max_iter=40000)

    model_list = [[svm, svm_hp, "svm"], [knn, knn_hp, "knn"], [log_reg, log_reg_hp, "logreg"]]

    for model in model_list:
        clf = GridSearchCV(model[0], model[1], cv=10)
        clf.fit(X_train, y_train)
        print("Model:", model[2])
        print("Best parameters set found on development set:")
        print(clf.best_params_)
        print("Grid scores on development set:")
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))

        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        best_models.append((clf.best_estimator_, clf.best_params_, clf.best_score_))

    max = 0
    best = ()
    for models in best_models:
        if models[2] > max:
            max = models[2]
            best = models

    return best

def plot(X, Y, name):

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.5, random_state=0)

    svm = sk.svm.LinearSVC(max_iter=40000)

    knn = sk.neighbors.KNeighborsClassifier()

    log_reg = sk.linear_model.LogisticRegression(solver='lbfgs', fit_intercept=True, max_iter=10000)


    graph = ValidationCurve(svm, param_name="C",
        param_range=[.01, .1, 1, 10], cv=10)
    graph.fit(X_train, y_train)
    title = name + "svm.png"
    graph.finalize(title=name + "SVM")
    graph.poof(outpath=title)

    #####

    g = ValidationCurve(knn, param_name="n_neighbors", param_range=[50, 75, 100, 125],
                        cv=10)
    g.fit(X_train, y_train)
    title = name + "knn.png"
    g.finalize(title=name + "knn")
    g.poof(outpath=title)

    #####

    g3 = ValidationCurve(log_reg, param_name="C",
                          param_range=[100, 10, 1, 0.1], cv=10)
    g3.fit(X_train, y_train)
    title = name + "logreg.png"
    g3.finalize(title = name + "logistic_regression")
    g3.poof(outpath=title)

#DRIVER
X, Y = word_embeddings('trainreviews.txt')

X = sk.preprocessing.normalize(X)
#plot(X,Y, "word_embedding_")
vec = vectorizer()
X_b, Y_b = bag_of_words(vec, test = False)
X_b = sk.preprocessing.normalize(X_b)
#plot(X_b, Y_b, "bag_o_word_")
print "Word embeddings"
best_we = model_select(X,Y)
print best_we
print "Bag of words"
best_bow = model_select(X_b,Y_b)
print best_bow
print "Bag of Words Logistic Regression with C = 10 wins:", best_bow
X_test = bag_of_words(vec, test = True)
my_best_clf = sk.linear_model.LogisticRegression(solver='lbfgs', fit_intercept=True, max_iter=10000).fit(X_b, Y_b)

preds = my_best_clf.predict(X_test)

C_L = [(.789 -(1.96 * .058)), (.789 +(1.96 * .058)) ]
print "\nMy confidence interval for the accuracy of this classifier is:", C_L

"""
# FOR TESTING DURING DEV
X_test = X[1500:]
X_train = X[:1500]
Y_test = Y[1500:]
Y_train = Y[:1500]




svm_accuracy = SVM(X_train,Y_train, X_test, Y_test)
knn_accuracy = KNN(X_train, Y_train, X_test, Y_test, knn_hyperparam)
log_reg_accuracy = LogReg(X_train,Y_train, X_test, Y_test)
print(svm_accuracy, knn_accuracy, log_reg_accuracy )
"""



