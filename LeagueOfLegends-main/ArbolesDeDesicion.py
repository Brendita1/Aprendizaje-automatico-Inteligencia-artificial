import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
import pandas as pd
import numpy as np
from sklearn import tree
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from matplotlib import pyplot
def AlgoritmoArbolesDeDesicion():
    #Leer datos
    pima = pd.read_csv("Lol.csv")
    pima.head()
    feature_cols = ['wins','losses','kills','deaths','assists','doubleKills','tripleKills','quadraKills','pentaKills','totalDamageDealt','magicDamageDealt','physicalDamageDealt','trueDamageDealt','largestCriticalStrike','totalDamageDealtToChampions','magicDamageDealtToChampions','physicalDamageDealtToChampions','trueDamageDealtToChampions','champLevel','totalMinionsKilled','goldEarned','goldSpent','totalMinionsKilled','neutralMinionsKilled'   ]
    X = pima[feature_cols].values
    y = pima['hotStreak'].values
    #Normalizar datos
    X = preprocessing.StandardScaler().fit(X).transform(X)
    #Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X,  y, test_size=0.3, random_state=1) # 70% training and 30% test
    #Entrenar modelo
    clf = DecisionTreeClassifier(max_depth = 3, 
                                 random_state = 0)
    clf=clf.fit(X_train, y_train)
     #Predecir
    y_pred = clf.predict(X_test)
     #Evaluar
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    print("Accuracy:",metrics.accuracy_score(y_train, clf.predict(X_train)))

    fn=['wins','losses','kills','deaths','assists','doubleKills','tripleKills','quadraKills','pentaKills','totalDamageDealt','magicDamageDealt','physicalDamageDealt','trueDamageDealt','largestCriticalStrike','totalDamageDealtToChampions','magicDamageDealtToChampions','physicalDamageDealtToChampions','trueDamageDealtToChampions','champLevel','totalMinionsKilled','goldEarned','goldSpent','totalMinionsKilled','neutralMinionsKilled'   ]
    cn=['True', 'False']
    fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
    tree.plot_tree(clf,
                   feature_names = fn, 
                   class_names=cn,
                   filled = True);
    fig.savefig('imagename.png')
    #Graficar
    ns_probs = [0 for _ in range(len(y_test))]
    lr_probs = clf.predict_proba(X_test)
    lr_probs = lr_probs[:, 1]
    ns_auc = roc_auc_score(y_test, ns_probs)
    lr_auc = roc_auc_score(y_test, lr_probs)
    # Imprimimos en pantalla
    print('Sin entrenar: ROC AUC=%.3f' % (ns_auc))
    print('Regresión Logística: ROC AUC=%.3f' % (lr_auc))
    # Calculamos las curvas ROC
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
    # Pintamos las curvas ROC
    pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='Sin entrenar')
    pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Regresión Logística')
    # Etiquetas de los ejes
    pyplot.xlabel('Tasa de Falsos Positivos')
    pyplot.ylabel('Tasa de Verdaderos Positivos')
    pyplot.show()        

    #Implementacion 

    X_new= [[12,34,12,34,50,300,4000,6000,600,600,3,12,4,15,67,12,3,4,5,6,12,33,56,78]]
    implementation= clf.predict(X_new)
    print(implementation)

