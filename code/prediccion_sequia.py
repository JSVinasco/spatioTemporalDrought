"""
Created on Mon Jan  6 15:13:25 2020

@author: juan

Script para la prediccion de la sequia usando RF
"""



#########################################
# definimos las funciones de grass format - numpy y viceversa

import numpy as np

from grass.pygrass.raster.buffer import Buffer
from grass.pygrass.gis.region import Region

def raster2numpy(rastname, mapset=''):
    """Return a numpy array from a raster map"""
    with RasterRow(rastname, mapset=mapset, mode='r') as rast:
        return np.array(rast)


def numpy2raster(array, mtype, rastname, overwrite=False):
    """Save a numpy array to a raster map"""
    reg = Region()
    if (reg.rows, reg.cols) != array.shape:
        msg = "Region and array are different: %r != %r"
        raise TypeError(msg % ((reg.rows, reg.cols), array.shape))
    with RasterRow(rastname, mode='w', mtype=mtype, overwrite=overwrite) as new:
        newrow = Buffer((array.shape[1],), mtype=mtype)
        for row in array:
            newrow[:] = row[:]
            new.put_row(newrow)


################################################################################
################################################################################

# importamos las librerias para el procesamiento


from grass.pygrass.raster import RasterRow
import matplotlib.pyplot as plt
import grass.temporal as tgis
from datetime import datetime

# realizamos la conexion con la base de datos temporal
tgis.init()
dbif = tgis.SQLDatabaseInterfaceConnection()
dbif.connect()


# creamos el strds que debemos rellenar
SPI_RF = 'spi_rf'
dataset = tgis.open_new_stds(name=SPI_RF, type='strds',
                             temporaltype='absolute',
                             title="SPI RF",
                             descr="SPI predicho por RF",
                             semantic='mean', overwrite=True)




dataset_name_rf = 'spi_rf@PERMANENT'
dataset = tgis.open_old_stds(dataset_name_rf, "strds",dbif=dbif)

SPI_XG = 'spi_xg'
dataset = tgis.open_new_stds(name=SPI_XG, type='strds',
                             temporaltype='absolute',
                             title="SPI XG",
                             descr="SPI predicho por XG",
                             semantic='mean', overwrite=True)




dataset_name_xg = 'spi_xg@PERMANENT'
dataset = tgis.open_old_stds(dataset_name_xg, "strds",dbif=dbif)


# abrimos los antiguos strds para el calculo

#Prec_Mensual_Acu
prec = 'Prec_Mensual_Acu'
prec_strds = tgis.open_old_stds(prec, "strds",dbif=dbif)
prec_strds.get_registered_maps(columns='name,start_time')
num_prec = len(prec_strds.get_registered_maps(columns='name,start_time'))
#dtdelta = datetime.timedelta(days = int(7))

#et_dineof
et_dineof = 'et_dineof@PERMANENT'
et_dineof_strds = tgis.open_old_stds(et_dineof, "strds",dbif=dbif)
et_dineof_strds.get_registered_maps(columns='name,start_time')
num_et_dineof = len(et_dineof_strds.get_registered_maps(columns='name,start_time'))

#evi_dineof
evi_dineof = 'evi_dineof'
evi_dineof_strds = tgis.open_old_stds(evi_dineof, "strds",dbif=dbif)
evi_dineof_strds.get_registered_maps(columns='name,start_time')
num_evi_dineof = len(evi_dineof_strds.get_registered_maps(columns='name,start_time'))

#lai_dineof
lai_dineof = 'lai_dineof@PERMANENT'
lai_dineof_strds = tgis.open_old_stds(lai_dineof, "strds",dbif=dbif)
lai_dineof_strds.get_registered_maps(columns='name,start_time')
num_lai_dineof = len(lai_dineof_strds.get_registered_maps(columns='name,start_time'))

#lst_dineof
lst_dineof = 'lst_dineof@PERMANENT'
lst_dineof_strds = tgis.open_old_stds(lst_dineof, "strds",dbif=dbif)
lst_dineof_strds.get_registered_maps(columns='name,start_time')
num_lst_dineof = len(lst_dineof_strds.get_registered_maps(columns='name,start_time'))

#nddi_dineof
nddi_dineof = 'nddi_dineof@PERMANENT'
nddi_dineof_strds = tgis.open_old_stds(nddi_dineof, "strds",dbif=dbif)
nddi_dineof_strds.get_registered_maps(columns='name,start_time')
num_nddi_dineof = len(nddi_dineof_strds.get_registered_maps(columns='name,start_time'))

#ndvi_dineof
ndvi_dineof = 'ndvi_dineof@PERMANENT'
ndvi_dineof_strds = tgis.open_old_stds(ndvi_dineof, "strds",dbif=dbif)
ndvi_dineof_strds.get_registered_maps(columns='name,start_time')
num_ndvi_dineof = len(ndvi_dineof_strds.get_registered_maps(columns='name,start_time'))

#ndwi_dineof
ndwi_dineof = 'ndwi_dineof@PERMANENT'
ndwi_dineof_strds = tgis.open_old_stds(ndwi_dineof, "strds",dbif=dbif)
ndwi_dineof_strds.get_registered_maps(columns='name,start_time')
num_ndwi_dineof = len(ndwi_dineof_strds.get_registered_maps(columns='name,start_time'))


'''
strds = ['Prec_Mensual_Acu@PERMANENT',
'et_dineof@PERMANENT',
'evi_dineof@PERMANENT',
'lai_dineof@PERMANENT',
'lst_dineof@PERMANENT',
'nddi_dineof@PERMANENT',
'ndvi_dineof@PERMANENT',
'ndwi_dineof@PERMANENT']
'''

################################################################################
################################################################################

# cargamos los modelos

import pickle

# load model from file
xgb = pickle.load(open("xgboost_guajira.pickle.dat", "rb"))
filename = 'rf_model2.sav'
regr = pickle.load(open(filename, 'rb'))

################################################################################
################################################################################

# iniciamos la prediccion

#num_prec
for i in range(num_ndwi_dineof):
    # precipitacion
    fec_prec = prec_strds.get_registered_maps(columns='name,start_time')[i][1]
    prec_raster= prec_strds.get_registered_maps(columns='name,start_time')[i][0]
    prec_map= np.nan_to_num(raster2numpy(prec_raster, mapset='PERMANENT'))
    #et
    fec_et = et_dineof_strds.get_registered_maps(columns='name,start_time')[i][1]
    et_raster= et_dineof_strds.get_registered_maps(columns='name,start_time')[i][0]
    et_map=  np.nan_to_num(raster2numpy(et_raster, mapset='PERMANENT'))
    #evi
    fec_evi = evi_dineof_strds.get_registered_maps(columns='name,start_time')[i][1]
    evi_raster= et_dineof_strds.get_registered_maps(columns='name,start_time')[i][0]
    evi_map=  np.nan_to_num(raster2numpy(evi_raster, mapset='PERMANENT'))
    #lai
    fec_lai = lai_dineof_strds.get_registered_maps(columns='name,start_time')[i][1]
    lai_raster= lai_dineof_strds.get_registered_maps(columns='name,start_time')[i][0]
    lai_map=  np.nan_to_num(raster2numpy(lai_raster, mapset='PERMANENT'))
    #lst
    fec_lst = lst_dineof_strds.get_registered_maps(columns='name,start_time')[i][1]
    lst_raster= lst_dineof_strds.get_registered_maps(columns='name,start_time')[i][0]
    lst_map=  np.nan_to_num(raster2numpy(lst_raster, mapset='PERMANENT'))
    #nddi
    fec_nddi = nddi_dineof_strds.get_registered_maps(columns='name,start_time')[i][1]
    nddi_raster= nddi_dineof_strds.get_registered_maps(columns='name,start_time')[i][0]
    nddi_map=  np.nan_to_num(raster2numpy(nddi_raster, mapset='PERMANENT'))
    #ndvi
    fec_ndvi = ndvi_dineof_strds.get_registered_maps(columns='name,start_time')[i][1]
    ndvi_raster= ndvi_dineof_strds.get_registered_maps(columns='name,start_time')[i][0]
    ndvi_map=  np.nan_to_num(raster2numpy(ndvi_raster, mapset='PERMANENT'))
    #ndwi
    fec_ndwi = ndwi_dineof_strds.get_registered_maps(columns='name,start_time')[i][1]
    ndwi_raster= ndwi_dineof_strds.get_registered_maps(columns='name,start_time')[i][0]
    ndwi_map=  np.nan_to_num(raster2numpy(ndwi_raster, mapset='PERMANENT'))
    # convertimos a una linea para predecir
    prec_l = np.reshape(prec_map,-1)
    et_l = np.reshape(et_map,-1)
    evi_l = np.reshape(evi_map,-1)
    lai_l = np.reshape(lai_map,-1)
    lst_l = np.reshape(lst_map,-1)
    nddi_l = np.reshape(nddi_map,-1)
    ndvi_l = np.reshape(ndvi_map,-1)
    ndwi_l = np.reshape(ndwi_map,-1)
    P = [prec_l,et_l,evi_l,lai_l,lst_l,nddi_l,ndvi_l,ndwi_l]
    P = np.asarray(P).T
    xgb_predicion = xgb.predict(P)
    rf_prediccion = regr.predict(P)
    xgb_predic = np.reshape(xgb_predicion,(1398, 1864))
    rf_predic = np.reshape(rf_prediccion,(1398, 1864))
    # le damos un nombre a nuestra prediccion
    xgb_nombre='spi_xgb2_'+str(i)+'_@PERMANENT'
    rf_nombre='spi_rf2_'+str(i)+'_@PERMANENT'
    numpy2raster(xgb_predic, mtype='FCELL', rastname=xgb_nombre, overwrite=True)
    numpy2raster(rf_predic, mtype='FCELL', rastname=rf_nombre, overwrite=True)
    fech=fec_prec
    fecha = fech.strftime("%Y") +'-'+fech.strftime("%m")+'-'+fech.strftime("%d")
    #tgis.register_maps_in_space_time_dataset(type='raster',name=dataset_name_rf,maps=xgb_nombre,start=fecha,interval=True,update_cmd_list=True)
    #tgis.register_maps_in_space_time_dataset(type='raster',name=dataset_name_xg,maps=rf_nombre,start=fecha,interval=True,update_cmd_list=True)
    #dataset.update_from_registered_maps()
    #dataset.print_shell_info()
