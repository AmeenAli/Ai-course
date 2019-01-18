from math import *
from hw3_utils import *
import numpy as np
import operator
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import csv

def euclidean_distance(x , y):
    return sqrt(sum((iter[0] - iter[1]) ** 2 for iter in list(zip(x , y))))



class knn_factory(abstract_classifier_factory):
    def __init__(self , k):
        self.k = k
    def train(self, data, labels):
        self.samples = data
        self.labels = labels
        knn = knn_classifier(self.samples , self.labels , self.k)
        return knn
class knn_classifier(abstract_classifier):
    def __init__(self , training_data ,labels ,   k):
        self.training_data = training_data
        self.k = k
        self.labs = labels
        # print(np.array(labels).shape)
    def getTag(self , testInstance):
        distances = {}
        Postive = 0
        Negative = 0
        counter = 1
        for x in range(len(self.training_data)):
            dist = euclidean_distance(testInstance, self.training_data[x])
            distances[x] = dist
            counter += 1
        instances_by_distance = sorted(distances.items(), key=operator.itemgetter(1))
        Neighbors = []
        for inst in range(self.k):
            Neighbors.append(instances_by_distance[inst][0])
        for x in range(len(Neighbors)):
            label = self.labs[Neighbors[x]]
            if label == False:
                Negative += 1
            if label == True:
                Postive += 1
        if Negative > Postive:
            return False
        return True
    def classifiyList(self , list_of_instances):
        Tags = []
        for data in list_of_instances:
            Tags.append(self.classify(data))
        return Tags

    def classify(self, features):
        return self.getTag(features[0])


def split_crosscheck_group(dataset, num_folds):
    data = dataset[0]
    labels = dataset[1]
    indices_zero = []
    indices_one = []
    for tmp in range(len(data)):
        if labels[tmp] == 0:
            indices_zero.append(tmp)
    for tmp in range(len(data)):
        if labels[tmp] == 1:
            indices_one.append(tmp)

    zeros = np.random.permutation(indices_zero)
    ones = np.random.permutation(indices_one)

    ZerosParts = [zeros[i: i + ceil(len(zeros) / num_folds)] for i in range(0, len(zeros), ceil(len(zeros) / num_folds))]
    OnesParts = [ones[i: i + ceil(len(ones) / num_folds)] for i in range(0, len(ones), ceil(len(ones) / num_folds))]
    splitted = [list(ZerosParts[fold]) + list(OnesParts[fold]) for fold in range(num_folds)]

    for i, index in enumerate(splitted):
        data_tmp = [data[j] for j in index]
        labels_tmp = [labels[j] for j in index]
        with open('ecg_fold_{}.data'.format(i + 1), 'wb') as myFile:
            pickle.dump((data_tmp, labels_tmp, None), myFile)


def load_k_fold_data(index):
    with open('ecg_fold_{}.data'.format(index), 'rb') as output:
        res = pickle.load(output)
    return res

def evaluate(classifier_factory, k):
    AccuracyTOTAL = 0
    ErrorTOTAL = 0
    for TestNum in range(1, k + 1):
        print('Fold  : %s' % str(TestNum))
        test_group_features, test_group_labels , _ = load_k_fold_data(TestNum)
        train_group_features = []
        train_group_labels = []
        for TrainFold in range(1, k + 1):
            if TrainFold != TestNum:
                features, labels , _ = load_k_fold_data(TrainFold)
                for tmp in features:
                    train_group_features.append(tmp)
                for tmp2 in labels:
                    train_group_labels.append(tmp2)
        classifier = classifier_factory.train(train_group_features, train_group_labels)
        hit = 0
        for feature, real_label in zip(test_group_features, test_group_labels):
            res = classifier.classify([feature])
            if res == real_label:
                hit += 1
        assert len(test_group_features) != 0
        accuracy = hit / len(test_group_features)
        AccuracyTOTAL += accuracy
        miss = 1 - accuracy
        ErrorTOTAL += miss
    assert k != 0
    avg_accuracy = AccuracyTOTAL / k
    avg_error = ErrorTOTAL / k
    return avg_accuracy, avg_error

#***************************************************

class ID3Classifier(abstract_classifier):

    def __init__(self, tree_classifier):
        self.tree_classifier = tree_classifier
    def classify(self, features):
        return self.tree_classifier.predict(features)


class ID3Factory(abstract_classifier_factory):
    def train(self, data, labels):
        classifier = DecisionTreeClassifier(criterion="entropy")
        classifier.fit(data, labels)
        return ID3Classifier(classifier)


class PerceptronClassifier():
    def __init__(self, percpetron_classifier):
        self.clf = percpetron_classifier
    def classify(self, feature_vector):
        return self.clf.predict(feature_vector)

class Perceptron_factory(abstract_classifier_factory):
    def train(self, data, labels):
        clf = Perceptron()
        clf.fit(data , labels)
        return PerceptronClassifier(clf)


#************************************************************************************#

class myClassifier():
    def __init__(self, percpetron_classifier , knn_class):
        self.clf = percpetron_classifier
        self.knn = knn_class
    def classify(self, feature_vector):
        return self.clf.predict(feature_vector) & self.knn.predict(feature_vector)

class MyClassFactory(abstract_classifier_factory):
    def train(self, data, labels):
        clf = RandomForestClassifier(n_estimators=95)
        knn = KNeighborsClassifier(n_neighbors=1)
        clf.fit(data , labels)
        knn.fit(data , labels)
        return myClassifier(clf , knn)




if __name__ == '__main__' :

    data = load_data()
    split_crosscheck_group(data ,2)
    kths = [1 , 3 , 5 , 7 , 13]

    Results = []
    for k in kths:
        knn = knn_factory(k)
        acc, err = evaluate(knn, 2)
        Results.append((k, acc, err))

    with open('experiments6.csv', "w+") as fCsvfile:
        wCsvwriter = csv.writer(fCsvfile)
        wCsvwriter.writerow([])
        wCsvwriter.writerows(Results)


    # # Question Number 7
    id3 = ID3Factory()
    Accuracy1 , Error1 = evaluate(id3 , 2)

    clf = Perceptron_factory()
    Accuracy2 , Error2 = evaluate(clf , 2)

    Results2 = [(1 , Accuracy1 , Error1) , (2 , Accuracy2 , Error2)]

    with open('experiments12.csv', "w+") as fCsvfile:
        wCsvwriter = csv.writer(fCsvfile)
        wCsvwriter.writerow([])
        wCsvwriter.writerows(Results2)


