#Ejercicio 5:

#Establecer directorio de trabajo:
setwd("/Users/jlsovaz/Desktop/curso_data_mining/TEORIA/UNIDAD 1/")

#Establecer semilla para lograr reproducibilidad en los resultados obtenidos:
set.seed(123)

#Importar el dataset del primer problema en formato .RData con load:
load("vinos.RData")

muestra <- sample(1:nrow(vinos), 40) 
entrenamiento <- vinos[muestra,]
prueba <- vinos[-muestra,]

#Ahora aplicamos el método SVM para construir el modelo, con la variable "var" 
#dependiente de todas las restantes:
library(e1071)
modelo <- svm(var ~ ., data = entrenamiento)

#Exploramos el modelo
summary(modelo)

#Ahora se puede utilizar el modelo para predecir a qué clase pertenecen los vinos del subset de entrenamiento, los mismos que se usaron para la construcción del modelo.
resultados.entrenamiento <- predict(modelo, newdata = entrenamiento, type = "class") 
table(resultados.entrenamiento, entrenamiento$var)
#Vemos que sólamente 2 vinos son clasificados incorrectamente.

#Podemos probar a predecir la clase de los vinos del subset de validación:
resultados.prueba <- predict(modelo, newdata = prueba, type = "class") 
t <- table(resultados.prueba, prueba$var)
t ; 100 * sum(diag(t)) / sum(t)

#En este caso, sólamente 1 vino fue clasificado incorrectamente como Godello.

