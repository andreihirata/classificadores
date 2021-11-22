#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exemplo para alguns classificadores usado em aula.

As features de entrada são as mesmas usadas no trabalho com k-NN

veja que os rótulos são a última coluna


"""
import sys
import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import tree

def main():        
        print ("Loading data...")
        tr = np.loadtxt('treinamento.txt')
        ts = np.loadtxt('teste.txt')
        (l,c) = tr.shape # em caso de treino e teste com diferente tamanhos, alterar isso.
        y_test = ts[:, c-1]
        y_train = tr[:, c-1]
        X_train = tr[:, 1 : c-1]
        X_test = ts[:, 1 : c-1]
        
 # k-NN classifier
        from sklearn.metrics import classification_report

        for i in range(1,15):
                neigh = KNeighborsClassifier(n_neighbors=i, metric='euclidean')
                neigh.fit(X_train, y_train)
                # neigh.score(X_test, y_test)
                # print('*********************k-NN************************')
                with open("knn_n{}.txt".format(i), "w") as oFile:
                        print(classification_report(y_test, neigh.predict(X_test)), file=oFile)
                        oFile.close()

# DT - Decision Tree
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X_train, y_train)
        # print('*********************DT************************')
        with open("DT.txt", "w") as oFile:
                print(classification_report(y_test, clf.predict(X_test)), file=oFile)
                oFile.close()


##SVM com Grid search
        for i in (-3, -2, -1, 0, 1, 2, 3):
                C_range =  2. ** np.arange(-5 + i, 15 + i, 2)
                gamma_range = 2. ** np.arange(3 + i, -15 + i, -2)
                k = [ 'rbf']
                # instancia o classificador, gerando probabilidades
                srv = svm.SVC(probability=True, kernel='rbf')
                ss = StandardScaler()
                pipeline = Pipeline([ ('scaler', ss), ('svm', srv) ])
                
                param_grid = {
                        'svm__C' : C_range,
                        'svm__gamma' : gamma_range
                }
        
                # faz a busca
                grid = GridSearchCV(pipeline, param_grid, n_jobs=-1, verbose=True)
                grid.fit(X_train, y_train)
        
                # recupera o melhor modelo
                model = grid.best_estimator_
                # print('*********************SVM************************')
                with open("SVM_Grid_{}.txt".format(i), "w") as oFile:
                        print(classification_report(y_test, model.predict(X_test)), file=oFile)
                        oFile.close()

        
        
    
        
## MLP
        for a in (1e-5, 1e-4, 1e-3):
             for l1 in (50, 75, 100, 125, 150):
                     for l2 in (50, 75, 100, 125, 150):
                             for l3 in (50, 75, 100, 125, 150):
                                     scaler = StandardScaler()
                                     X_train = scaler.fit_transform(X_train)
                                     X_test = scaler.fit_transform(X_test)
        
                                     clf = MLPClassifier(solver='adam', alpha=a, hidden_layer_sizes=(l1, l2, l3), random_state=1)
                                     clf.fit(X_train, y_train)
                                     #print(clf.predict(X_test))
                                     # print('*********************MLP************************')
                                     with open("MLP_{}_{}_{}_{}.txt".format(a, l1, l2, l3), "w") as oFile:
                                             print(classification_report(y_test, clf.predict(X_test)), file=oFile)
                                             oFile.close()

## Random Forest Classifier
        for e in (7500, 10000, 125000):
                for d in (15, 30, 45):
                        #X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0, random_state=0, shuffle=False)
                        clf = RandomForestClassifier(n_estimators=e, max_depth=d, random_state=1)
                        clf.fit(X_train, y_train)  
                        # print(clf.feature_importances_)
                        # print(clf.predict(X_test))
                        # print('*********************Random Forest************************')
                        with open("RFC_{}_{}.txt".format(e, d), "w") as oFile:
                                print(classification_report(y_test, clf.predict(X_test)), file=oFile)
                                oFile.close()
                        
        
if __name__ == "__main__":
        if len(sys.argv) != 1:
                sys.exit("Classifiers.py")

        main()
