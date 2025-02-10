#Ejercicio 6:

#APRIORI

#Establecer directorio de trabajo:
setwd("/Users/jlsovaz/Desktop/curso_data_mining/TEORIA/UNIDAD 6/")

#Cargacmos el paquete necesario para poder utilizar el método Apriori:
library(arules)

#Cargamos el dataset que vamos a utilizar:
load("titanic.RData")

#Es necesario transformar el dataset a analizar en un objeto de clase "transactions". Para ello:
transacciones <- as(titanic, "transactions")

#Aplicamos el método apriori cambiando el parámetro de confianza a 0.9 (0.8 es el valor por defecto)
reglas <- apriori(transacciones, parameter = list(support = 0.1, confidence = 0.80))

#Ordenamos las reglas de mayor a menor confianza:
reglas.ordenadas <- sort(reglas, by = "lift")

#Transformamos el objeto reglas ordenadas en data.frame:
reglas.ordenadas.dataframe <- as(reglas.ordenadas, "data.frame")

#Exploramos el resultado:
reglas.ordenadas.dataframe

#El algoritmo ha encontrado 12 subconjuntos frecuentes, aquellos con frecuencia o support mayor o igual que 0,1 
#y con confianza de la regla al menos 0,90. Los subconjuntos y reglas están ordenados por importancia decreciente.
#La importancia la podemos definir en función de la frecuencia (support) o la importancia (confidence)

#El primer subconjunto frecuente que encontramos (con la mayor confidencia) es el que se define por tres condiciones:
#sexo = mujer, empleo = inactivo. La regla establece la edad >= 55 como resultado de las otras dos condiciones.
#Tiene una frecuencia absoluta de 103 casos sobre 1000

#El lift es una medida de calidad de la regla. Un valor próximo a 1 indica que ambos lados de la regla son independients
#Si es mayor que 1, indica que ambos lados están correlacionados. Por ejemplo, un valor de lift de 3 significa
#que la parte derecha de la regla ocurre con una frecuencia 3 veces mayor de lo que cabría esperar si no 
#estuviera relacionada con la parte izquierda.

#Estas reglas indican relaciones dentro del conjunto de datos. En este caso, parece que las mujeres inactivas son mayores
#los compradores inactivos o los inactivos en general son mayores, los hombres casados de edad entre 35-55 son activos, etc.



#PAGERANK

#Cargamos el paquete necesario para poder realizar el algoritmo de PageRank:
library(igraph)

#Establecer directorio de trabajo:
setwd("/Users/jlsovaz/Desktop/curso_data_mining/TEORIA/UNIDAD 6/")

#Cargamos el dataset con el que trabajaremos:
Dataset <- read.csv2("futbol.csv", header=TRUE,encoding="latin1")

#Es neceario convertir el dataframe que queremos analizar en un objeto "grafo"
grafo <- graph.data.frame(Dataset)

#Establecemos una semilla para la reproducibilidad de los resultados:
set.seed(123)

#Ploteamos el grafo para visualizar todas las relaciones de i conoce a j
plot(grafo)

#Calcular el índice de PageRank:
I <- page.rank(grafo)
I
rev(sort(I$vector)) #Ordenar de forma decreciente

#La persona más conocida, en términos ponderados es P, seguida de B, aunque 
#ésta es conocida directamente por más personas (11 frente a 9 de P como veíamos 
#en las frecuencias relativas). El más relevante (P) no es necesariamente el 
#más conocido (B), sino el que es más conocido por aquellos que a su vez 
#son muy conocidos.

#Ploteamos de nuevo y redimensionamos los nodos en función del índice de PageRank:
#Aquellos nodos más grandes son más importantes en relación a PageRank:
plot(grafo, vertex.size=I$vector*200)
