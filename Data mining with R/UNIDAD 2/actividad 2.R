########ADABOOST########


library(adabag)

setwd("/Users/jlsovaz/Desktop/curso_data_mining/TEORIA/UNIDAD 2/")

set.seed(123)

load("empresas.RData")

#El banco está interesado en predecir los clientes que tendrán impagos basándose en esa información diponible.

#Dividimos los datos en dos submuestras. Una de ellas, de tamaño 80, será utilizada para construir el clasificador, 
#y la otra, con los 20 casos restantes, la utilizaremos para validar los resultados: 
#se predice la clase de cada elemento de la muestra de validación con las funciones 
#generadas sin utilizar los datos de ese elemento.

muestra_entrenamiento <- sample(1:nrow(empresas), 1000) #Cogemos aleatoriamente 80 filas (clientes) de nuestro dataset inicial
subset_entrenamiento <- empresas[muestra_entrenamiento,]
subset_validacion <- empresas[-muestra_entrenamiento,]

#la fórmula que relaciona la variable dependiente con las variables predictoras, 
#y el conjunto de datos utilizado “entrenamiento”. El símbolo ~ que indica relación lineal 
modelo_boosting <- boosting(CULTURAL ~ ., data = subset_entrenamiento)
#Variable dependiente: Impago (es como queremos clasificar a nuestros clientes, en función de una serie de variables: relación lineal)
#el conjunto de predictores por un punto, que representa a todas las variables excepto la dependiente:
#modelo <- boosting(Impago ~ ., data = entrenamiento)

#El algoritmo genera 100 reglas o clasificadores débiles con forma de árbol de decisión, que se pueden listar con:
modelo_boosting$trees
#La función crea 100 clasificadores débiles, vemos el último. Crea 5 nodos. 
#El nodo 1 clasifica de 80 clientes, 39 mal (probabilidad de acierto 51.23%)

modelo_boosting$weights
modelo_boosting$class
modelo_boosting$prob
modelo_boosting$importance #La suma de % da 100%. Vemos que Deud_ing e Ingresos presentan la mayor importancia. Son los que contribuyen en mayor medida a la clasificación

#Ahora vamos a predecir a que clase irán los datos de la muestra de validación.
#Podemos utilizar el modelo para predecir la clase de cualquier elemento o caso 
#con la función predict. Por ejemplo, aplicada a todos los elementos del conjunto “entrenamiento” (que es el conjunto que utilizamos para construir el modelo)
predicción_entrenamiento <- predict(modelo_boosting, newdata = subset_entrenamiento, type = "class")

#predicción_entrenamiento$confusion: es la matriz que cruza la clasificación de la predicción con la clasificación del modelo
predicción_entrenamiento$confusion
#Vemos que hace una clasificación completamente perfecta con respecto al modelo (como era de esperar)

#Sin embargo, la cosa cambia cuando intentamos predecir muestras nuevas (no usadas para construir el modelo):
predicción_validacion <- predict(modelo_boosting, newdata = subset_validacion, type = "class")

predicción_validacion$confusion

#Expresar el % de acierto:
100 * sum(diag(predicción_validacion$confusion)) / sum(predicción_validacion$confusion) # calcula el porcentaje global de acierto
#Suma de los datos que están en la diagonal entre el total: 19/20 = 95%


#Como no hay problema de sobreajuste (es decir, clasifica bien la muestra de entrenamiento y la de validación) podemos aplicar el boosting a la muestra total deudas (100 muestras)
modelo_boosting_empresas <- boosting(CULTURAL ~ ., data = empresas)

#Intentamos predecir la clasificación de las muestras de deudas (que son las mismas que fueron usadas para la creación del modelo)
prediccion_empresas <- predict(modelo_boosting_empresas, newdata = empresas, type = "class") 

prediccion_empresas$confusion

#% de acierto
100 * sum(diag(prediccion_empresas$confusion)) / sum(prediccion_empresas$confusion)
#100%





###############RANDOM FOREST###################

library(randomForest)

setwd("/Users/jlsovaz/Desktop/curso_data_mining/TEORIA/UNIDAD 1/")

load("algas.RData")

set.seed(123)

muestra <- sample(1:nrow(algas), 20) # 40 números de fila o caso elegidos al azar 
entrenamiento <- algas[muestra, ]
prueba <- algas[-muestra, ]

modeloRF <- randomForest(clase~ ., data=entrenamiento)
modeloRF


predicciones <- predict(modeloRF, prueba)
t <- with(prueba, table(predicciones, clase)) 
t ; 100 * sum(diag(t)) / sum(t)


modeloRF_algas <- randomForest(clase~ ., data=algas)
predicciones <- predict(modeloRF_algas, algas)
t <- with(algas, table(predicciones, clase)) 
t ; 100 * sum(diag(t)) / sum(t)







