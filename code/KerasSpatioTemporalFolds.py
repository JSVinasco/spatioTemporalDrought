#Se importan las librerías
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.regularizers import l2
from keras import optimizers
import pylab
import gdal
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import math



# establecemos un directorio de trabajo
os.chdir('E:/0_Tesis/trn_csv/')

os.getcwd()

# obtenemos los archivos para trabajar
ruta = 'E:/0_Tesis/trn_csv/'


# leemos nuestro archivo
df = pd.read_csv(ruta+'trn_completa_3m.csv',sep=',',index_col=False,parse_dates=["fecha"])
df = df.iloc[:,[0,1,2,3,10,11,12,13,14,15,16,17]]
cols_to_norm = ['SPI_3_meses','ET', 'EVI', 'LAI','LST', 'NDDI', 'NDVI', 'NDWI', 'Prec_Mensual']
df[cols_to_norm] = df[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))



# hacemos que R pueda ver el DataFrame
r_df = pandas2ri.py2ri(df)
type(r_df)
#calling function under base package
print(base.summary(r_df))


# creamos ST n_folds
CSTF = CAST.CreateSpacetimeFolds(r_df,spacevar='estacion',timevar='fecha')
#print(pandas2ri.ri2py(CSTF.names))
temp = pandas2ri.ri2py(CSTF)


################################################################################


# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(df.iloc[:,4:12], df.iloc[:,3], test_size = 0.15)

x_train = np.asarray(x_train)
x_test = np.asarray(x_test)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

################################################################################
# y nos aproximaremos a una evaluacion del modelo

# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.title('model mean squared error')
plt.ylabel('root mean squared error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()


es_546 =  df['estacion']==531
'''1632, 1724, 1725, 1726, 1730, 1731, 1732, 1733, 1734, 1735, 1736,
       1737, 1738, 1740, 1741, 1742, 1743, 1744, 1746, 1747, 1751,  250,
         39,   40,   45,   50,  504,  505,  507,  510,  515,  517,  519,
        522,  529,  530,  531,  532,  535,  536,  537,  538,  539,  541,
        543,  544,  546]'''
es_546 = df[es_546]
predictores_546 = es_546.iloc[:,4:12]
result_546 = es_546.iloc[:,3]
prediccion_546 = model.predict(np.asarray(predictores_546))
coefficient_of_dermination = metrics.r2_score(result_546, prediccion_546)


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

################################################################################
#
#       finalmente aplicamos la validacion
#
###############################################################################


oos_y = []
oos_pred = []

for i in range(len(temp)):
    for j in range(len(temp[1])):
        x_train = np.asarray(df.iloc[list(temp[0][j][:-1]),4:12])
        y_train= np.asarray(df.iloc[list(temp[0][j][:-1]),3])
        x_test = np.asarray(df.iloc[list(temp[1][j][:-1]),4:12])
        y_test = np.asarray(df.iloc[list(temp[1][j][:-1]),3])
        model = Sequential()
        model.add(Dense(11, input_dim=8, activation='relu',kernel_regularizer=l2(0.0001)))
        model.add(Dropout(0.05))
        model.add(Dense(4,  activation='relu',kernel_regularizer=l2(0.0001)))
        model.add(Dropout(0.05))
        #model.add(Dense(11, activation='relu',kernel_regularizer=l2(0.0001)))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mean_absolute_error',
                        optimizer='Adagrad',
                        metrics=['accuracy'])
        history = model.fit(x_train,
                            y_train,
                            validation_data=(x_test,y_test),
                            verbose=0,
                            epochs=100,
                            shuffle=True,
                            batch_size=32)
        #model.fit(x_train,y_train,validation_data=(x_test,y_test),verbose=0,epochs=500, shuffle=True)
        pred = model.predict(x_test)
        oos_y.append(y_test)
        oos_pred.append(pred)
        # Measure this fold's RMSE
        score = np.sqrt(metrics.mean_squared_error(pred,y_test))
        print(f"Fold score (RMSE): {score}")




from sklearn.metrics import r2_score

R2 = []
estacion = []
for i in df.estacion.unique():
    es_546 =  df['estacion']==i
    es_546 = df[es_546]
    predictores_546 = es_546.iloc[:,4:12]
    result_546 = es_546.iloc[:,3]
    prediccion_546 = model.predict(np.asarray(predictores_546))
    coefficient_of_dermination = r2_score(result_546, prediccion_546)
    R2.append(coefficient_of_dermination)
    estacion.append(str(i))

H2 = np.asarray([R2,estacion]).T
p2 = pd.DataFrame(H2, columns=['Coeficiente de Correlacion','Estacion'])
p2





R2 = []
estacion = []
for i in df.estacion.unique():
    es_546 =  df['estacion']==i
    es_546 = df[es_546]
    predictores_546 = es_546.iloc[:,4:12]
    result_546 = es_546.iloc[:,3]
    prediccion_546 = model.predict(np.asarray(predictores_546))
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



#############################################################################

import tensorflow as tf
import pandas as pd
import os
import numpy as np
from sklearn import metrics
#import keras
#from keras.layers import Dropout
#from scipy.stats import zscor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
import scipy.stats
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
#from tensorflow.keras.layers import LeakyReLU
#from tensorflow.keras import optimizers
#from keras.layers import Dropout
### primero entrenaremos una red individual
j=0
oos_y = []
oos_pred = []
#x_train = np.asarray(df.iloc[list(temp[0][j][:-1]),4:12])
#y_train= np.asarray(df.iloc[list(temp[0][j][:-1]),3])
#x_test = np.asarray(df.iloc[list(temp[1][j][:-1]),4:12])
#y_test = np.asarray(df.iloc[list(temp[1][j][:-1]),3])

model = Sequential()
model.add(Dense(11, input_dim=8, kernel_initializer='random_uniform',bias_initializer='zeros',activation='relu',kernel_regularizer=l2(0.001)))
model.add(Dropout(0.05))
model.add(Dense(11, kernel_initializer='random_uniform', bias_initializer='zeros',activation='relu',kernel_regularizer=l2(0.001)))
model.add(Dropout(0.05))
#model.add(Dense(11, activation='relu',kernel_regularizer=l2(0.0001)))
model.add(Dense(1, activation='linear'))
## Define optimizer: Stochastic gradient descent
#gd = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#sgd = keras.optimizers.Adam(learning_rate=0.01, beta_1=0.09, beta_2=0.99, amsgrad=False)
#sgd = keras.optimizers.Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
#'Adagrad'
#sgd = keras.layers.LeakyReLU(alpha=0.3)
model.compile(loss='mse',
                optimizer=Adagrad(learning_rate=0.0001),
                metrics=[tf.keras.metrics.RootMeanSquaredError(),'accuracy'])
# checkpoint
#filepath="weights.best.hdf5"
#checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#callbacks_list = [checkpoint]
#train
history = model.fit(x_train,
                    y_train,
                    validation_data=(x_test,y_test),
                    verbose=0,
                    epochs=1000,
                    shuffle=True,
                    batch_size=25)
pred = model.predict(x_test)
oos_y.append(y_test)
oos_pred.append(pred)
# Measure this fold's RMSE
score = np.sqrt(metrics.mean_squared_error(pred,y_test))
print(f"Fold score (RMSE): {score}")

