from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn import preprocessing
import matplotlib.pyplot as plt
def AlgoritmoRegresionLinealMultiple():
    #Leer datos
    pima = pd.read_csv("Lol.csv")
    pima.head()
    feature_cols = ['wins','losses','item0','item1','item2','item3','item4','item5','item6','deaths','assists','doubleKills','tripleKills','quadraKills','pentaKills','unrealKills','totalDamageDealt','magicDamageDealt','physicalDamageDealt','trueDamageDealt','largestCriticalStrike','totalDamageDealtToChampions','magicDamageDealtToChampions','physicalDamageDealtToChampions','trueDamageDealtToChampions','champLevel','totalMinionsKilled','damageDealtToObjectives','largestMultiKill','killingSprees','totalHeal','totalUnitsHealed','damageSelfMitigated','damageDealtToObjectives','damageDealtToTurrets','totalDamageTaken','magicalDamageTaken','physicalDamageTaken','trueDamageTaken','totalMinionsKilled','goldEarned','goldSpent','totalMinionsKilled','neutralMinionsKilled','baronKills','dragonKills','riftHeraldKills']   
    X = pima[feature_cols].values
    y = pima['kills'].values
    #Normalizar datos
    X = preprocessing.StandardScaler().fit(X).transform(X)
    #Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    #Entrenar modelo
    lr_multiple = linear_model.LinearRegression()
    lr_multiple.fit(X_train, y_train)
    #Predecir
    Y_pred_multiple = lr_multiple.predict(X_test)

    print('DATOS DEL MODELO REGRESIÓN LINEAL MULTIPLE')
    print()
    print('Valor de las pendientes o coeficientes "a":')
    print(lr_multiple.coef_)
    #Evaluar
    print('Precisión del modelo:')
    print(lr_multiple.score(X_train, y_train))
    print('Precisión del modelo:')
    print(lr_multiple.score(X_test, y_test))

    plt.scatter(pima['goldEarned'].values, pima['kills'].values,  color='blue')
    plt.xlabel("goldEarned")
    plt.ylabel("kills")
    plt.show()
    plt.scatter(y_test, Y_pred_multiple,  color='blue')
    plt.xlabel("Prueba")
    plt.ylabel("Prediccion")
    plt.show()