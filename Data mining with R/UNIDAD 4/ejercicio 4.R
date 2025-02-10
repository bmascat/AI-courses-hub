
#En primer lugar, seleccionamos nuestro directorio de trabajo. En mi caso:
  setwd("/Users/jlsovaz/Desktop/curso_data_mining/TEORIA/UNIDAD 0/")
  
#Importamos el dataset del primer problema en formato .RData con la función load:
  load("deudas.RData")
  
#Establecemos una semilla para lograr reproducibilidad en los resultados obtenidos:
  set.seed(123)

#Construimos la muestra que servirá de entrenamiento con 40 casos y la de validación con los restantes:
  muestra <- sample(1:nrow(deudas), 80) 
  entrenamiento <- deudas[muestra, ]
  prueba <- deudas[-muestra, ]

  #Cargamos el paquete necesario para aplicar el método CART:
  library(C50)

  #Aplicamos el método C50
  modelo <- C5.0(Impago ~ Ingreso+Deud_ing+Deud_tarj+Deud_otr, data = entrenamiento)
  
  #Exploramos el modelo construído:
summary(modelo)  
  
#Construímos la matriz de confusión con el subset de validación:
resultados.entrenamiento <- predict(modelo, newdata = entrenamiento, type = "class") 
table(resultados.entrenamiento, entrenamiento$Impago)

modelo <- C5.0(Impago ~ Ingreso+Deud_ing+Deud_tarj+Deud_otr, data = deudas)
resultados.entrenamiento <- predict(modelo, newdata = deudas, type = "class") 
table(resultados.entrenamiento, deudas$Impago)

plot(modelo)

  