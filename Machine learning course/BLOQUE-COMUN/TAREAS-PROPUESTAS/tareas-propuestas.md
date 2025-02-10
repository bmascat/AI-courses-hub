# Tareas propuestas

El alumno puede elegir una o proponer una tarea equivalente al profesor.

## 1) Etiquetación morfosintáctica con NeuroNLP2 o NCRFpp

El alumno deberá usar NCRFpp o NeuroNLP2 para realizar pruebas sobre etiquetación sintáctica.

**Corpora**: se puede seleccionar los que se consideren convenientes, pero se aconseja usar los 
corpora
de [Universal Dependencies](https://universaldependencies.org) que tengan un mínimo de,
aproximadamente, 400 mil palabras sumando en los conjuntos de entrenamiento/validación/test.

**Posibles pruebas** (se debe elegir una o proponer una alternativa):

- Obtener los resultados (*Accuracy*) obtenidos por una misma configuración de NeuroNLP2 o NCRFpp 
sobre al menos 4 corpora, preferiblemente de varios idiomas distintos.
- Probar al menos 2 configuraciones para los bloques constituyentes de los modelos, al menos sobre 
2 corpora cada 1. También se pueden probar 4 o mas configuraciones sobre el mismo corpus.
  - En el caso de NeuroNLP2, se puede elegir si usar un CRF o no como capa final para hacer la
  predicción (parámetro `"crf"` del archivo de configuración), y el tipo de red recurrente usada 
  en el bloque de procesamiento de la entrada (parámetro `"rnn_mode"`, que puede tomar los valores 
  `RNN`, `GRU`, `LSTM` y `FastLSTM`). También se puede especificar el número de capas recurrentes a 
  usar (parámetro "num_layers").
  
  - En el caso de NCRpp, se puede seleccionar si usar un bloque CRF para la predicción (parámetro
  `use_crf`), si usar *embeddings* de caracteres (parámetro `use_char`), el tipo de red recurrente
  o convolucional a usar para el procesamiento de palabras (`word_seq_feature`) o caracteres
  (`char_seq_feature`). Estos dos ultimos casos permiten una mayor variabilidad que en el caso de
  NeuroNLP2, ya que podemos elegir entre `CNN`, `LSTM` o `GRU` tanto para caracteres como 
  para palabras. También podemos elegir el número de capas a usar tanto con una `CNN` (parámetro 
  `cnn_layer`) o como una una `LSTM` (parámetro `lstm_layer`).
  
  - El alumno también puede hacer pruebas configurando otros parámetros: usar diferentes
   `word embeddings` para el mismo lenguaje; usar diferentes tamaños para los `char embeddings`;
   usar diferentes valores de `dorpout`; etc

- Optimizar los hiperparámetros para una misma configuración de NeuroNLP2 o NCRFpp sobre 
al menos un corpus. El procedimiento para ello es, dado un conjunto de valores posibles para cada 
parámetro, entrenar modelos con cada una de las combinaciones de valores posibles, seleccionando la 
que de mejores resultados. El número de modelos a entrenar es el producto de las cardinalidades de 
los conjuntos de valores a testear para cada parámetro: |valores param 1| x |valores param 2 x 
|valores param 3| x ...
  
El alumno deberá entregar una pequeña memoria (al menos 3 páginas en PDF) resumiendo el proceso de 
entrenamiento, e indicando los corpora usados, las configuraciones y pruebas realizadas, y los 
resultados obtenidos (concretamente la métrica de *Accuracy*). Con respecto a estos últimos, se 
pueden indicar los resultados sobre el conjunto de desarrollo (dev) o prueba (test), o incluír 
también curvas de aprendizaje indicando el progreso del mismo a lo largo del entrenamiento: el eje 
corresponde a la iteración, y el eje Y a la *Accuracy* obtenida en dicha iteración.

Se valorará el número de tests realizados, y la calidad de la memoria aportada.

<div style="page-break-before: always;"></div>

## 2) Reconocimiento de Entidades Nombradas con NeuroNLP2 o NCRFpp

El alumno deberá usar NCRFpp o NeuroNLP2 para realizar pruebas sobre Reconocimiento de Entidades 
Nombradas.

**Corpora**: en principio se puede usar el corpus en inglés de 
[CONLL-03](https://www.clips.uantwerpen.be/conll2003/ner/), pero se pueden obtener corpora 
adicionales [en la siguiente página](https://github.com/juand-r/entity-recognition-datasets), o
[esta 
otra](https://lionbridge.ai/datasets/15-free-datasets-and-corpora-for-named-entity-recognition- 
ner/) .

Nota: no todos los corpora listados en los dos últimos enlace están disponibles o son gratuítos. 
Además, no todos están en formato CoNLL-03 o el aceptado por las herramientas, por lo que puede ser 
necesario convertirlos. Una tarea que el alumno puede incluír en la memoria es la descripción de 
donde se obtuvieron estos recursos y como fueron procesados.

**Posibles pruebas** (se debe elegir una o proponer una alternativa):

- Obtener los resultados (*Precision*, *Recall* y *F1-score*) obtenidos por una misma configuración 
de NeuroNLP2 o NCRFpp sobre al menos 3 corpora, preferiblemente de varios idiomas distintos.
- Probar al menos 2 configuraciones para los bloques constituyentes de los modelos, al menos sobre 
2 corpora cada uno. También se pueden probar 4 o mas configuraciones sobre el mismo corpus.
  - En el caso de NeuroNLP2, se puede elegir si usar un CRF o no como capa final para hacer la
  predicción (parámetro `"crf"` del archivo de configuración), y el tipo de red recurrente usada 
  en el bloque de procesamiento de la entrada (parámetro `"rnn_mode"`, que puede tomar los valores 
  `RNN`, `GRU`, `LSTM` y `FastLSTM`). También se puede especificar el número de capas recurrentes a 
  usar (parámetro "num_layers").
  
  - En el caso de NCRpp, se puede seleccionar si usar un bloque CRF para la predicción (parámetro
  `use_crf`), si usar *embeddings* de caracteres (parámetro `use_char`), el tipo de red recurrente
  o convolucional a usar para el procesamiento de palabras (`word_seq_feature`) o caracteres
  (`char_seq_feature`). Estos dos ultimos casos permiten una mayor variabilidad que en el caso de
  NeuroNLP2, ya que podemos elegir entre `CNN`, `LSTM` o `GRU` tanto para caracteres como 
  para palabras. También podemos elegir el número de capas a usar tanto con una `CNN` (parámetro 
  `cnn_layer`) o como una una `LSTM` (parámetro `lstm_layer`).
  
  - El alumno también puede hacer pruebas configurando otros parámetros: usar diferentes
   `word embeddings` para el mismo lenguaje; usar diferentes tamaños para los `char embeddings`;
   usar diferentes valores de `dropout`; etc


- Optimizar los hiperparámetros para una misma configuración de NeuroNLP2 o NCRFpp sobre 
al menos un corpus. El procedimiento para ello es, dado un conjunto de valores posibles para cada 
parámetro, entrenar modelos con cada una de las combinaciones de valores posibles, seleccionando la 
que de mejores resultados. El número de modelos a entrenar es el producto de las cardinalidades de 
los conjuntos de valores a testear para cada parámetro: |valores param 1| x |valores param 2 x 
|valores param 3| x ...

El alumno deberá entregar una pequeña memoria (al menos 3 páginas en PDF) resumiendo el proceso de 
entrenamiento, e indicando los corpora usados, las configuraciones y pruebas realizadas, y los 
resultados obtenidos (concretamente las métricas de *Precision*, *Recall* y *F1-score*). Con 
respecto a estos últimos, se pueden indicar los resultados sobre el conjunto de desarrollo (dev) o 
prueba (test), o incluír también curvas de aprendizaje indicando el progreso del mismo a lo largo 
del entrenamiento: el eje corresponde a la iteración, y el eje Y la *Precision/Recall/F1-score* 
obtenida en dicha iteración.

Se valorará el número de tests realizados, los corpora utilizados y la calidad de la memoria 
aportada.

<div style="page-break-before: always;"></div>

## 3) Análisis Sintáctico de Dependencias con NeuroNLP2

El alumno deberá usar NeuroNLP2 para realizar pruebas sobre Análisis Sintáctico de Dependencias.

**Corpora**: se puede seleccionar los que se consideren convenientes, pero se aconseja usar los 
corpora de [Universal Dependencies](https://universaldependencies.org) que tengan un mínimo de,
aproximadamente, 400 mil palabras sumando en los conjuntos de entrenamiento/validación/test.

**Posibles pruebas** (se debe elegir una o proponer una alternativa):

- Obtener los resultados (*UAS* y *LAS*) obtenidos por una misma configuración de NeuroNLP2 sobre 
al menos 4 corpora, preferiblemente de varios idiomas distintos.
- Probar al menos 2 configuraciones para los bloques constituyentes de los modelos, al menos sobre 
2 corpora.

  - Se puede elegir el tipo de analizador a usar (parámetro `"model"`, que puede tomar los valores 
  `DeepBiAffine`, `NeuroMST` o `StackPtr`), y el tipo de red recurrente usada en el bloque de
  procesamiento de la entrada (parámetro `"rnn_mode"`, que puede tomar los valores `RNN`, `GRU`,
 `LSTM` y `FastLSTM`). También se puede especificar el número de capas recurrentes a 
  usar (parámetro "num_layers").

  - El alumno también puede hacer pruebas configurando otros parámetros: usar diferentes
   `word embeddings` para el mismo lenguaje; usar diferentes tamaños para los `char embeddings`;
   usar diferentes valores de `dorpout`; etc

- Optimizar los hiperparámetros para una misma configuración de NeuroNLP2 o NCRFpp sobre 
al menos un corpus. El procedimiento para ello es, dado un conjunto de valores posibles para cada 
parámetro, entrenar modelos con cada una de las combinaciones de valores posibles, seleccionando la 
que de mejores resultados. El número de modelos a entrenar es el producto de las cardinalidades de 
los conjuntos de valores a testear para cada parámetro: |valores param 1| x |valores param 2 x 
|valores param 3| x ...
 
El alumno deberá entregar una pequeña memoria (al menos 3 páginas en PDF) resumiendo el proceso de 
entrenamiento, e indicando los corpora usados, las configuraciones y pruebas realizadas, y los 
resultados obtenidos (concretamente las métricas de *UAS* u *LAS*). Con respecto a estos últimos, 
se pueden indicar los resultados sobre el conjunto de desarrollo (dev) o prueba (test), o incluír 
también curvas de aprendizaje indicando el progreso del mismo a lo largo del entrenamiento: el eje 
corresponde a la iteración, y el eje Y al resultado de la métrica obtenida en dicha iteración.

Se valorará la calidad de la memoria y el número y complejidad de las pruebas realizadas.

<div style="page-break-before: always;"></div>

## 4) Clasificación de texto con Keras y TensorFlow, para un dataset preprocesado.

El alumno deberá escribir, usando Keras y TensorFlow, una herramienta de clasificación de texto.

**Dataset**: Se utilizará el *Reuters-21578 Dataset*. Tiene la ventaja de que se puede 
cargar directamente desde [Keras](https://keras.io/api/datasets/reuters/):

```
     from keras.datasets import reuters
     from keras.preprocessing.text import Tokenizer
   
     ...

     (x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=None, test_split=0.2)

```

Al final de este código, `x_train` y `x_test` son arrays bidimensionales en los que cada entrada es 
un texto, y las palabras de cada texto aparecen codificadas como enteros. Por su parte, `y_train` e 
`y_test` son arrays de enteros de una sola dimensión, en los que cada entrada (etiqueta) corresponde 
a la clase a la que pertenece el texto. Si quereis ver las palabras del primer texto de `x_train`, 
podeis usar el siguiente código:

```
word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded = ' '.join([reverse_word_index.get(i - 3, '?') for i in x_train[0]])
print(decoded)
```

A partir de aquí podemos procesar la entrada de varias maneras. Vamos a considerar tres:

**Posibilidad 1**. A partir de aquí, podemos recodificar los textos de `x_train` y `x_test` en forma 
de arrays binarios, en los que cada posición corresponde a una palabra del vocabulario contenido en 
`word_index`, y tendrá valor 1 si la palabra en cuestión aparece en el texto (0 en caso contrario):

```
from keras.preprocessing.text import Tokenizer

max_words = 20000  # el vocabulario del dataset es 30979 palabras. Nos quedamos con las 20000
                   # con mayor fecuencia

tokenizer = Tokenizer(num_words=max_words)
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
```

Con esta codificación, podeis hacer modelos de capas `Dense` o similares con 
`input_shape=(max_words,)` para la primera capa. La última capa de la red debería ser `Dense`, con 
un número de unidades igual al número de clases en `y_train` (`max(y_train) + 1`, en python) y 
activación `'softmax'`.

**Posibilidad 2**. Otra alternativa es dejar `x_train` y `x_test` como están, para entrenar una 
capa `Embedding` a la entrada del modelo. Para ello, primero tendríamos que igualar las longitudes 
de las entradas, añadiendo ceros al final:

```
max_sequence = 250 #damos la misma longitud a todas las entradas de x_train e x_test

x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_sequence,
                                                        padding='post', truncating='post')
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_sequence,
                                                       padding='post',truncating='post')
```

En este caso, el parámetro `input_dim` de la capa `Embedding` sería `max_words`, mientras que 
`input_length` sería `max_sequence`. Dependiendo de si estais usando un modelo `Sequential` o la 
API funcional, sería:

```
modelo.add(Embedding(max_words, 128, input_length=max_sequence)))
```
o

```
capa_embedding = Embedding(max_words, 128, input_length=max_sequence))
``` 

**Posibilidad 3**. Podeis usar *embeddings* preentrenados. Teneis un ejemplo de como hacerlo en 
[este
tutorial](https://keras.io/examples/nlp/pretrained_word_embeddings/). La idea es obtener el 
vocabulario del corpus, descargar embeddings para el mismo idioma (en el tutorial se usan los de 
[GloVe](https://nlp.stanford.edu/projects/glove/)), y crear una matriz con los embeddings de dicho 
vocabulario, con la que inicializar la capa `Embedding` (a través del parámetro 
`embeddings_initializer`).

<div style="page-break-before: always;"></div>

Con respecto a las etiquetas de los ejemplos de entrenamiento o test, podemos convertir `y_train` 
e `y_test` a vectores one-hot para poder usar `'categorical_crossentropy'` como función de error:

```
num_classes = max(y_train) + 1
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
```
o simplemente usar `'sparse_categorical_crossentropy'`.

Si usamos *embeddings* de palabras, tanto no entrenados como preentrenados, a continuación de la 
capa `Embedding`, se deben añadir más capas para el procesamiento de las secuencias de entrada. 
Para ello, se pueden usar simples capas `Dense`, capas de redes convolucionales (`Conv1D`, 
`GlobalMaxPooling1D`, `GlobalAveragePooling1D`) o capas recurrentes (`SimpleRNN`, `RNN`, `GRU`, 
`LSTM`, `Bidirectional`). Dado que se trata de un problema de clasificación no binaria, la última 
capa de la red debería ser `Dense`, con `num_classes` como número de unidades y activación 
`'softmax'`. Es aconsejable usar capas `Dropout` para minimizar el sobreentrenamiento.

Si se usa más de una capa recurrente (GRU, LSTM, RNN, Bidirectional) la segunda y sucesivas 
capas RNN tienen que recibir como entrada el estado de la capa anterior en todos los 
instantes de tiempo, no sólo al final de la secuencia. Ello se consigue con `return_sequences=True` 
como parámetro al definir la primera capa. Teneis un ejemplo de ello, precisamente sobre 
clasificación de texto, en 
[este enlace](https://www.tensorflow.org/tutorials/text/text_classification_rnn).

**Pruebas**: Comparar el rendimiento de al menos dos redes de neuronas, usando una o más de las 
configuraciones propuestas, aunque al menos una de ellas debe usar embeddings. Para medir los 
resultados se usará la métrica *Accuracy*.

El alumno deberá entregar una pequeña memoria, explicando las redes desarrolladas, incluyendo el 
código de las mismas, y los resultados obtenidos. Se puede entregar en PDF o como una libreta 
Jupyter.

Se valorará el número de pruebas realizadas y la complejidad de las redes desarrolladas.

<div style="page-break-before: always;"></div>


## 5) Clasificación de texto con Keras y TensorFlow, para un dataset no preprocesado.

El alumno deberá examinar el tutorial sobre análisis de sentimiento en comentarios de la IMDB, [con
Keras](https://www.tensorflow.org/tutorials/keras/text_classification), e implementar el ejercicio 
que viene al final, para hacer predicciones de etiquetas sobre preguntas de [Stack 
Overflow](https://stackoverflow.com).

**Pruebas**: Comparar el rendimiento de al menos dos redes de neuronas para este problema. Para 
medir los resultados se usará la métrica *Accuracy*.

El alumno deberá entregar una pequeña memoria, explicando las redes desarrolladas, incluyendo el 
código de las mismas, y los resultados obtenidos. Se puede entregar en PDF o como una libreta 
Jupyter.

Se valorará el número de pruebas realizadas y la complejidad de las redes desarrolladas.

