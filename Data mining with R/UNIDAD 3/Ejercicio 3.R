#Para poder aplicar CHAID, necesario instalar y cargar el paquete "CHAID"
library(CHAID)

#Con chaid_control configuramos todos los parámetros del algoritmo, aunque lo que vienen por defecto son los adecuados.
chaid_control(alpha2 = 0.05, alpha3 = -1, alpha4 = 0.05,minsplit = 20, minbucket = 7, 
              minprob = 0.01,stump = FALSE, maxheight = -1)

#Establecemos directorio de trabajo:
setwd("/Users/jlsovaz/Desktop/curso_data_mining/TEORIA/UNIDAD 1/")

#Cargamos el dataset compra
load("deudas.RData")

summary(deudas)

#Tenemos dos variables que son ordinales: edad y formación. Tenemos que ordenar los niveles correctamente (por defecto, los niveles están ordenados alfabéticamente)
deudas$Edad <- ordered(deudas$Edad, levels=c("menos de 30", "entre 30 y 40", "mas de 40"))
deudas$Formacion <- ordered(deudas$Formacion, levels=c("elemental", "bachillerato", "estudios universitarios"))
deudas$Empleo <- ordered(deudas$Empleo, levels=c("menos de 5 a\xf1os", "entre 5 y 10 a\xf1os", "m\xe1s de 10 a\xf1os"))
deudas$Residencia <- ordered(deudas$Residencia, levels=c("menos de 5 a\xf1os", "entre 5 y 10 a\xf1os", "m\xe1s de 10 a\xf1os"))

q1 <- quantile(deudas$Ingreso,0.33) 
q3 <- quantile(deudas$Ingreso,0.67)
deudas$Ingreso <- recode(deudas$Ingreso, 'lo:q1="bajo"; q1:q3="medio"; q3:hi="alto" ', as.factor=TRUE)
deudas$Ingreso <- ordered(deudas$Ingreso, levels=c('bajo' ,  'medio', 'alto'))

q1 <- quantile(deudas$Deud_ing,0.33) 
q3 <- quantile(deudas$Deud_ing,0.67)
deudas$Deud_ing <- recode(deudas$Deud_ing, 'lo:q1="bajo"; q1:q3="medio"; q3:hi="alto" ', as.factor=TRUE)
deudas$Deud_ing <- ordered(deudas$Deud_ing, levels=c('bajo' ,  'medio', 'alto'))

q1 <- quantile(deudas$Deud_tarj,0.33) 
q3 <- quantile(deudas$Deud_tarj,0.67)
deudas$Deud_tarj <- recode(deudas$Deud_tarj, 'lo:q1="bajo"; q1:q3="medio"; q3:hi="alto" ', as.factor=TRUE)
deudas$Deud_tarj <- ordered(deudas$Deud_tarj, levels=c('bajo' ,  'medio', 'alto'))

q1 <- quantile(deudas$Deud_otr,0.33) 
q3 <- quantile(deudas$Deud_otr,0.67)
deudas$Deud_otr <- recode(deudas$Deud_otr, 'lo:q1="bajo"; q1:q3="medio"; q3:hi="alto" ', as.factor=TRUE)
deudas$Deud_otr <- ordered(deudas$Deud_otr, levels=c('bajo' ,  'medio', 'alto'))

summary(deudas)


#Aplicamos el método CHAID, configuramos algunos parámetros antes
chaid_control(alpha2 = 0.05, alpha3 = -1, alpha4 = 0.05,minsplit = 20, minbucket = 7, 
              minprob = 0.01,stump = FALSE, maxheight = -1)
ch <- chaid(Impago ~ ., data = deudas) #Utilizando una relación lineal con todas las variables cualitativas usando ".,"

ch #Muestra los resultados
plot(ch) #Dibuja el árbol de clasificación

#Vemos que la variable explicativa más significativa para clasificar a las personas en relación con la variable "comprador" es el estado civil.
#Divide entonces por estado civil. Luego se emplean las siguientes variables para seguir realizando separaciones de grupo.
#En este caso, parece ser que la variable empleo no parece ser significativa y no se utiliza para separar grupos.

#El árbol también nos da información sobre la proporción de personas en relación a compradores o no compradores.
#La mayor parte de los compradores son casados menores de 35 años (70%), mientras que los no compradores son
#no casados, mujeres, mayores de 55 años con formación universitaria (50%). Así podemos analizar cada grupo en función de las variables cualitativas.




#Naive Bayes
#Cargamos el paquete necesario para aplicar el método Naive Bayes:
library(e1071)

#Se supone que los predictores cuantitativos siguen una distribución normal.

#Cargamos el dataset "pacientes"
load("algas.RData")

#En este dataset, recodificamos la variable hba1c a una cualitativa con dos niveles: hasta 7, mayor que 7
#La función recode es cargada por la libreria car
library(car)
pacientes$hba1c7 <- recode(pacientes$hba1c, ' lo:7="hasta 7"; 7:hi="mayor que 7" ', as.factor=T)

#Ahora aplicaremos el método Naive Bayes para intentar relacionar esta variable que tiene dos 
#niveles con otras variables del conjunto, utilizadas como predictoras.

#Vamos a filtrar el dataset sólamente cogiendo las variables necesarias para el análisis (hay variables muy correlacionadas o no necesarias)
datos = pacientes[ , c('edad', 'sexo', 'alcohol', 'tabaco', 'dieta', 'peso', 'talla', 'tad', 'tas', 'colesterol', 'pericintura', 'peripelvis', 'trigl', 'creat', 'alcoholgrdia','imc', 'icc', 'hta2', 'hba1c7')]

#A partir de este nuevo dataset, creamos nuestras dos muestras: la de entrenamiento y la de validación:
set.seed(12345)
muestra <- sample(1:nrow(algas), 20) 
entrenamiento <- algas[muestra,]
prueba <- algas[-muestra,]

#Aplicamos el método:
modelo <- naiveBayes(x = entrenamiento, y = algas[muestra,"clase"])

modelo
#Globalmente, el grupo de hba1c7 “hasta 7” tiene el 74,7% de los casos, y "mayor que 7" el 25,3% restante.

#La edad es una variable numérica. En el grupo “hasta 7” la media de edades es 59,83 años y la desviación típica 10,9 años. 
#Para la variable sexo, cualitativa, se muestra la probabilidad (frecuencia) de hombres y mujeres en cada grupo.


#Establecemos la predicción, con los datos del conjunto de validación:
resultados <- predict(object = modelo, newdata = prueba, type = "class") 
t <- table(resultados, algas[-muestra,"clase"])
t ; 100 * sum(diag(t)) / sum(t)

#Los resultados son solamente discretos: las variables explicativas permiten predecir la clase correcta solamente en el 70,99% de los casos.

#Ahora añadiremos un paciente nuevo, el cual tenemos toda la información de las diferentes variables predictoras, pero no conocemos su clase en relación a la variable hba1c7
#Lo añadimos al dataset marcando un NA en la variable hba1c7
datos[882,] <- c(62, "mujer", "si", "no", 0, 92, 1.62, 100, 160, 263, 104, 112, 116, 0.9, 8, 35.05563, 0.9285714, "si", NA)

#Ahora aplicamos la función predict para poder predecir la clase de este nuevo paciente en relación al modelo de NaiveBayes:
predict(modelo, datos[882,], type = "class") #Predice que su clase es "hasta 7" (con un 70% de probabilidad de acierto).

#Podemos entonces mejorar el modelo y aplicar el método a todo el dataset:
modelo <- naiveBayes(x = algas, y = algas[,"clase"])

modelo

#Establecemos la predicción, con los datos del conjunto de validación:
resultados <- predict(object = modelo, newdata = algas, type = "class") 
t <- table(resultados, algas[,"clase"])
t ; 100 * sum(diag(t)) / sum(t)
