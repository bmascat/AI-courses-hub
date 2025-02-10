#Ejercicio 1:

#Establecer directorio de trabajo:
setwd("/Users/jlsovaz/Desktop/curso_data_mining/TEORIA/UNIDAD 1/")

#Establecer semilla para lograr reproducibilidad en los resultados obtenidos:
set.seed(123)

#Importar el dataset del primer problema en formato .RData con load:
load("vinos.RData")

#Realizar método k-means en el dataframe "vinos" utilizando dos clases para clasificar
#cada muestra en función de la información que nos dan sobre las concentraciones de los
#distintos ácidos orgánicos. Elegimos dos clases porque es la clasificación "natural"
#de nuestras muestras en este caso: Albariño y Godello. Así veremos si las variables
#cuantitativas son un buen criterio de clasificación de Albariño y Godello.
kmeans_vinos <- kmeans(vinos[,2:ncol(vinos)], centers = 2)

#Exploramos el modelo calculado:
kmeans_vinos

#El tamaño para cada cluster es de: 28 vinos para una clase y 26 vinos para otra clase.


table(kmeans_vinos$cluster, vinos$var)


#############################################################

#Ejercicio 2:

#Establecer directorio de trabajo:
setwd("/Users/jlsovaz/Desktop/curso_data_mining/TEORIA/UNIDAD 0/")

#Establecer semilla para lograr reproducibilidad en los resultados obtenidos:
set.seed(123)

#Importar el dataset del primer problema en formato .RData con load:
load("deudas.RData")

#Realizar método EM en el dataframe "deudas" utilizando dos clases para clasificar
#cada muestra en función de la información que nos dan sobre ingresos, deudas_ingresos, deudas_tarjeta, otras deudas.
#Así veremos si las variables cuantitativas son un buen criterio de clasificación de los clientes en dos categorías: presentan o no presentan impagos.

#Cargamos el paquete necesario para realizar el método EM:
library(mclust)

#Realizamos nuestro modelo EM con dos clases:
EM_banco <-Mclust(deudas[,5:8], G=2)

#Exploramos nuestro modelo y vemos como clasificación se relaciona con la variable Impago:
EM_banco$classification
table(EM_banco$classification, deudas$Impago)





