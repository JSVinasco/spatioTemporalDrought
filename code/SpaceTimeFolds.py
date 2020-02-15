
### Este es un codigo para embeber la libreria la funcion
### CreateSpacetimeFolds del paquete CAST en una analisis
### realizado en python

#########################################
###   seccion 1: library importing    ###
########################################
#importamos las librerias

import pandas as pd
import os
import numpy as np
import math
#

# this project use a r2py project like a prerequisite 

import rpy2
from rpy2.robjects.packages import importr
#
import rpy2.robjects as robjects
base = importr('base')
CAST = importr('CAST')
utils = importr('utils')
from rpy2.robjects import pandas2ri

################################################
###   seccion 2: data reading and scaling    ###
################################################

# establecemos un directorio de trabajo
os.chdir('E:/0_Tesis/trn_csv/')

# obtenemos los archivos para trabajar
ruta = 'E:/0_Tesis/trn_csv/'

# leemos nuestro archivo
#from rpy2.robjects import pandas2ri # install any dependency package if you get error like "module not found"
pandas2ri.activate()

#robjects.DataFrame.from_csvfile('file="E:/0_Tesis/trn_csv/trn_completa_3m.csv",header=TRUE, sep=","')
df = pd.read_csv(ruta+'trn_completa_3m.csv',sep=',',index_col=False,parse_dates=["fecha"])
df = df.iloc[:,[0,1,2,3,10,11,12,13,14,15,16,17]]

########################################
###   seccion 3: space time folds    ###
########################################
r_df = pandas2ri.py2ri(df)
type(r_df)
#calling function under base package
print(base.summary(r_df))

# Aplicamos la funcion CreateSpacetimeFolds
CSTF = CAST.CreateSpacetimeFolds(r_df,spacevar='estacion',timevar='fecha')


# convertir a objetos python

from rpy2.robjects import pandas2ri
print(pandas2ri.ri2py(CSTF.names))
temp = pandas2ri.ri2py(CSTF)
print(temp[0])
print(temp[1])


############################################################
###   seccion 4: RF training with CV space-time folds    ###
############################################################


from sklearn.ensemble import RandomForestRegressor

mse = []
mae = []
mape = []
precision = []

for i in range(len(temp)):
    for j in range(len(temp[1])):
        X = np.asarray(df.iloc[list(temp[0][j][:-1]),4:12])
        y= np.asarray(df.iloc[list(temp[0][j][:-1]),3])
        #X, y  = make_regression(n_features=8, n_informative=2,
                              # random_state=0, shuffle=True)
        regr = RandomForestRegressor(n_estimators=350,n_jobs=-1, oob_score = True,warm_start=False)
        regr.fit(X, y)
        X_valid = np.asarray(df.iloc[list(temp[1][j][:-1]),4:12])
        predicciones = regr.predict(np.asarray(df.iloc[list(temp[1][j][:-1]),4:12]))
        test = np.asarray(df.iloc[list(temp[1][j][:-1]),3])
        errors = abs(predicciones - test)
        e = round(np.mean(errors),2)
        e2 = np.sqrt(e)
        #print('Error medio absoluto', round(np.mean(errors),2),'sequia')
        mae.append(e)
        mse.append(e2)
        #mape_ = 100*(errors / test)
        #mape.append(mape_)
        #precision_ = 100 - np.mean(mape)
        #precision.append(precision_)
        print_score(regr,X,y,X_valid,test)



H = np.asarray([mae,mse]).T
p = pd.DataFrame(H, columns=['MAE','MSE'])
p


#### save the model
import pickle
# guardamos nuestro ModelCheckpoint
filename = 'rf_model2.sav'
pickle.dump(regr, open(filename, 'wb'))



#######################################################
###   seccion 5: graph model perfomance evaluate    ###
#######################################################



es_546 =  df['estacion']==1632
es_546 = df[es_546]
predictores_546 = es_546.iloc[:,4:12]
result_546 = es_546.iloc[:,3]
prediccion_546 = regr.predict(predictores_546)

import matplotlib.pyplot as plt
plt.scatter(result_546,prediccion_546)
plt.title('Correlación entre los valores predichos y los reales')
plt.xlabel('Valores reales')
plt.ylabel('Valores predichos')
plt.show()


periodos = pd.date_range(start ='2001-01-31',  end ='2012-12-31', freq ='M')

plt.plot(periodos,result_546,'k')
plt.plot(periodos,prediccion_546, 'g')
plt.title('Comportamiento temporal del SPI en la locación 546')
plt.xlabel('Tiempo')
plt.ylabel('SPI no parametrico')
plt.show()

# grafico total

from sklearn.metrics import r2_score

R2 = []
estacion = []
for i in df.estacion.unique():
    es_546 =  df['estacion']==i
    es_546 = df[es_546]
    predictores_546 = es_546.iloc[:,4:12]
    result_546 = es_546.iloc[:,3]
    prediccion_546 = regr.predict(predictores_546)
    coefficient_of_dermination = r2_score(result_546, prediccion_546)
    R2.append(coefficient_of_dermination)
    estacion.append(str(i))
    plt.scatter(result_546,prediccion_546)
    plt.title('Correlación entre los valores predichos y los reales')
    plt.xlabel('Valores reales')
    plt.ylabel('Valores predichos')
    plt.show()
    plt.plot(periodos,result_546,'k')
    plt.plot(periodos,prediccion_546, 'g')
    plt.title('Comportamiento temporal del SPI en la locación 546')
    plt.xlabel('Tiempo')
    plt.ylabel('SPI no parametrico')
    plt.show()

H2 = np.asarray([R2,estacion]).T
p2 = pd.DataFrame(H2, columns=['Coeficiente de Correlacion','Estacion'])
p2


################################################################################
# XGBoosts
import xgboost as xgb
from xgboost import plot_importance

from sklearn.metrics import mean_squared_error

mse_r = []
mae_r = []
mape_r = []
precision_r = []

for i in range(len(temp)):
    for j in range(len(temp[1])):
        X = np.asarray(df.iloc[list(temp[0][j][:-1]),4:12])
        y= np.asarray(df.iloc[list(temp[0][j][:-1]),3])
        #X, y  = make_regression(n_features=8, n_informative=2,
                              # random_state=0, shuffle=True)
        #xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
        #                max_depth = 5, alpha = 10, n_estimators = 10)
        xg_reg = xgb.XGBRegressor(objective ='reg:squarederror',
                colsample_bytree=0.4,
                 gamma=0,
                 learning_rate=0.07,
                 max_depth=3,
                 min_child_weight=1.5,
                 n_estimators=10000,
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.6,
                 seed=42)
        xg_reg.fit(X,y)
        X_valid = np.asarray(df.iloc[list(temp[1][j][:-1]),4:12])
        predicciones = xg_reg.predict(X_valid)
        test = np.asarray(df.iloc[list(temp[1][j][:-1]),3])
        errors = abs(predicciones - test)
        e = round(np.mean(errors),2)
        e2 = np.sqrt(e)
        #print('Error medio absoluto', round(np.mean(errors),2),'sequia')
        mae_r.append(e)
        mse_r.append(e2)
        #print_score(xg_reg,X,y,X_valid,test)
        rmse = np.sqrt(mean_squared_error(test, predicciones))
        print("RMSE: %f" % (rmse))


H = np.asarray([mae_r,mse_r]).T
p = pd.DataFrame(H, columns=['MAE','MSE'])
p

###
es_546 =  df['estacion']==1632
es_546 = df[es_546]
predictores_546 = es_546.iloc[:,4:12]
result_546 = es_546.iloc[:,3]
prediccion_546 = xg_reg.predict(np.asarray(predictores_546))

import matplotlib.pyplot as plt
plt.scatter(result_546,prediccion_546)
plt.title('Correlación entre los valores predichos y los reales')
plt.xlabel('Valores reales')
plt.ylabel('Valores predichos')
plt.show()


periodos = pd.date_range(start ='2001-01-31',  end ='2012-12-31', freq ='M')

plt.plot(periodos,result_546,'k')
plt.plot(periodos,prediccion_546, 'g')
plt.title('Comportamiento temporal del SPI en la locación 546')
plt.xlabel('Tiempo')
plt.ylabel('SPI no parametrico')
plt.show()  


# save model to file
import pickle
pickle.dump(xg_reg, open("xgboost_guajira.pickle.dat", "wb"))

# load model from file
loaded_model = pickle.load(open("pima.pickle.dat", "rb"))

