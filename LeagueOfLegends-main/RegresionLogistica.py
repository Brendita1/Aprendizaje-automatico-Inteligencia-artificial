import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt 
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot

def Algoritmo_RegresionLogistica():
    
    #Leer datos
    df = pd.read_csv('Lol.csv')
    df.head()                                                                                                                               
    columns=['wins','losses','item0','item1','item2','item3','item4','item5','item6','kills','deaths','assists','doubleKills','tripleKills','quadraKills','pentaKills','unrealKills','totalDamageDealt','magicDamageDealt','physicalDamageDealt','trueDamageDealt','largestCriticalStrike','totalDamageDealtToChampions','magicDamageDealtToChampions','physicalDamageDealtToChampions','trueDamageDealtToChampions','champLevel','totalMinionsKilled','damageDealtToObjectives','largestMultiKill','killingSprees','totalHeal','totalUnitsHealed','damageSelfMitigated','damageDealtToObjectives','damageDealtToTurrets','totalDamageTaken','magicalDamageTaken','physicalDamageTaken','trueDamageTaken','totalMinionsKilled','goldEarned','goldSpent','totalMinionsKilled','neutralMinionsKilled','baronKills','dragonKills','riftHeraldKills']       
    X = df[columns].values
    y = df['hotStreak'].values

    #Normalizar datos
    X = preprocessing.StandardScaler().fit(X).transform(X)

    #Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=3)

    #Entrenar modelo
    LR = LogisticRegression(C=0.0001,solver='liblinear').fit(X_train, y_train)
    LR.coef_


    #Predecir
    yhat = LR.predict(X_test)


    print (yhat [0:20])
    print (y_test [0:20])

    #Evaluar
    print("Test Accuracy:", accuracy_score(y_test, yhat))
    print("Train test", accuracy_score(y_train, yhat))
    print("Confusion matrix:\n", confusion_matrix(y_test, yhat, labels=[1,0]))
    print ("Classification report:\n",classification_report(y_test, yhat))

    #Realizar grafico
    ns_probs = [0 for _ in range(len(y_test))]
    lr_probs = LR.predict_proba(X_test)
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
