# NCRF++

## Descripción

NCRF++ es una herramienta similar a NeuroNLP2, pero un poco más pulida y con más opciones respecto al tipo de redes de neuronas que implementa, pero sólo para problemas de etiquetación de secuencias, como reconocimiento de entidades nombradas o etiquetación morfosintáctica. Por lo tanto, no incluye análisis sintáctico. Al igual que NeuroNLP2, la arquitectura que implementa está basada en tres bloques: una capa para procesamiento de la caracteres (de la entrada), otra capa para el procesamiento de palabras, y una tercera para realizar la inferencia y generar la salida. La principal diferencia está en el tipo de redes neuronales implementadas por ambas herramientas:

  * NeuroNLP2 utiliza sólo Redes Convolucionales (CNNs) para el procesamiento de caracteres, pero permite selecccionar entre LSTMs, GRUs o RNNs de Elman para el procesamiento de palabras. Respecto al tercer bloque, las únicas opciones son un CRF para tareas de etiquetación de secuencias (que es opcional, en cuyo caso la salida de la red sería la salida del segundo bloque) o una capa de análisis sintáctico de dependencias.
  * NCRF++ permite elegir entre CNN, GRU o LSTM o nada para el procesamiento de caracteres, CNN, GRU o LSTM para el procesamiento de palabras, y CRF o Softmax para la generación de la salida de la red. En total hay 4x3x2=24 combinaciones posibles para la arquitectura de una red generada por NCRF++.
  
[La página de GitHub de NCRF++](https://github.com/jiesutd/NCRFpp)

## Instalación

Requerimientos: versión 3.6 o superior de Python, PyTorch y Gensim. Con `pip` se pueden instalar con los siguientes comandos:

```
pip3 install overrides   (este lo necesité en alguna instalación, pero es posible que no os haga falta)

pip3 install fasttext
pip3 install gensim

pip3 install torch torchvision
```

He copiado en la cuenta "visitante" de idefix, en el directorio `~/Taggers/NCRFpp/` lo siguiente:

  - 'Design Challenges and Misconceptions in Neural Sequence Labeling.pdf' es el pdf con el paper en donde se describe la herramienta y se hacen algunas pruebas de su funcionamiento. Entre otras, se usa para etiquetación morfosintáctica sobre el subcorpus WSJ del Penn Treebank, y se evalúan 12 de las posibles arquitecturas que puede generar la herramienta (todas las que no usan redes basadas en GRUs). 
  - `NCRFpp-master/` es el árbol de directorios de NCRF++. En general no va a ser necesario explorar mucho ese directorio. Salvo que se quiera introducir alguna modificación a los scripts para que generen una salida específica, lo único que vais a utilizar el es script `NCRFpp-master/main.py` para ejecutar la herramienta. En cualquier caso, tenemos los siguientes subdirectorios:
      - `model/` para la implementación de los modelos de red usados en la herramientoa.
      - `readme/` información sobre la herramienta.
      - `sample_data/` ficheros de ejemplo para la tarea de reconocimiento de entidades y embeddings.
      - `utils/` algunos scripts con utilidades.
      
      Además hay ejemplos de ficheros de configuración, para entrenamiento de una red (`demo.train.config`) y para usar una red entrenada sobre texto (`demo.decode.config`).
  - `experiments/` es un árbol de directorios que he creado similar al que teníamos en NeuroNLP2, para hacer los experimentos. Contiene los siguientes subdirectorios:
      - `configs/` para los archivos de configuración.
      - `data/` para los embeddings y corpora que se an a usar.
      - `models/` para los modelos y resultados generados por la herramienta.
      - `scripts/` para los scripts de bash para ejecutar los experimentos.

    La estructura de esos subdirectorios es la misma que he usado en NeuroNLP2, para que podais encontrar las cosas más fácilmente. Sin embargo, vosotros podeis configurar vuestros directorios como querais.
    
He dejado un script `experiments/scripts/run_pos_anCora-es.sh` para ejecutar la herramienta como etiquetador morfosintáctico sobre el corpus AnCora, un archivo de configuración en `experiments/configs/pos/run_pos_anCora-es.sh` y los embeddings y archivos de entrenamiento/validación y test en `experiments/data/cc.es.300.vec` y `experiments/data/pos/anCora-es/` respectivamente. También he dejado los archivos análogos para ejecutar la herramienta como un reconocedor de entidades nombradas sobre los conjuntos de datos de CoNLL-03 en inglés: `experiments/scripts/run_ner_conll.sh`, `experiments/configs/ner/conll.train.config`, y `experiments/data/cc.en.300.vec` y `experiments/data/ner/conll/`.

## Archivos de entrada

Respecto a los archivos con los datos de entrenamiento/desarrollo/prueba para etiquetación de secuencias, el formato es una palabra por línea, acompañada por la etiqueta correspondiente, separadas por un espacio en blanco, con una línea en blanco como separador de frases. Por ejemplo, para etiquetación morfosintáctica, lo mínimo sería:

```
Pierre NNP
Vinken NNP
, ,
61 CD
years NNS
old JJ
...
```

Para reconocimiento de entidades nombradas se podría hacer lo mismo con las etiquetas correspondientes.

```
Pierre B-PER
Vinken I-PER
, O
61 O
years O
old O
...
```

NCRF++ permite añadir características a los ejemplos de los ficheros de entrada (entrenamiento,validación,prueba). Por ejemplo, supongamos que queremos hacer reconocimiento de entidades y queremos añadir dos características:

* Si la palabra comienza por mayúsculas (que llamaremos `Cap`, de *capitalized*)
* La etiqueta morfosintáctica de la palabra (que llamaremos `POS`, de *POS*)

El resultado sería:

```
Pierre [Cap]1 [POS]NNP B-PER
Vinken [Cap]1 [POS]NNP I-PER
, [Cap]0 [POS], O
61 [Cap]0 [POS]CD O
years [Cap]0 [POS]NNS O
old [Cap]0 [POS]JJ O
...
```

Es fácil generar los archivos de entrenamiento de NCRF++ para etiquetación morfosintáctica a partir de los archivos en formato CoNLL-X que usa NeuroNLP2: basta con quedarse con las columnas 2 (la palabra) y 5 (la etiqueta) y descartar todo lo demás. En Unix o Linux se puede hacer directamente desde la linea de comandos de linux usando el comando cut. Ej:

```
cut -f 2,5 es_ancora-ud-dev.conllx --output-delimiter=" " > es_ancora-ud-dev.txt
```

donde `es_ancora-ud-dev.txt` sería el archivo de entrenamiento para NCRF++.

Con respecto al reconocimiento de entidades nombradas, he dejado en `~/Taggers/NCRFpp/experiments/scripts/` un script de python `CoNLL2NCRFpp.py` para trasformar los archivos de entrenamiento de CoNLL-03 que se pueden descargar de [aqui](https://github.com/glample/tagger/tree/master/dataset). Para ver las opciones del mismo ejecutad el comando `python3 CoNLL2NCRFpp -h`. Sale la siguiente información:

```
USAGE: python CoNLL2NCRFpp.py [-h] -s sourceFile -d destFile [-p] [-c]
ARGUMENTS:
  -h:           print this help
  -s:           path to the file from CoNLL-03
  -d:           path to the NCRFpp-compatible file
  -p:           add POS as a feature for NCRFpp (default=False)
  -c:           add Capitalization as a feature for NCRFpp (default=False)

```

En el caso de los embeddings de palabras, NCRF++ acepta archivos de texto **sin** comprimir, con una palabra por línea, con la palabra al principio de la línea, seguida de los elementos del embedding separados por espacios. Por lo tanto, para usar los embeddings de FastText, teneis que hacer dos pasos adicionales respecto a NeuroNLP2:

  - descomprimir los archivos de embeddings (en linux con gunzip).
  - borrar la primera línea del archivo, que contiene el tamaño del vocabulario y la dimensión de los embeddings.

Para usar la herramienta hay que ejecutar el script NCRFpp-master/main.py. En principio, ese script tiene algunas opciones de línea de comandos, pero para parámetros que se pueden especificar en el archivo de configuración (y en el archivo de configuración se pueden especificar más cosas que en la línea de comandos). Por lo tanto, he optado por usar el archivo de configuración para especificar todas opciones de la herramienta, y he dejado el path a dicho archivo de configuración como única opción de línea de comandos. Así, en el script experiments/scripts/run_pos_wsj.sh, el único comando es:

```
python3 ../NCRFpp-master/main.py --config configs/pos/wsj/wsj.train.config > models/pos/wsj/results-wsj.txt
```
(se supone que será ejecutado desde el directorio ~/Taggers/NCRFpp/experiments/, de ahí el path para llegar a main.py)

Hay un archivo de configuración con valores por defecto en NCRFpp-master/demo.train.config.

## Archivo de configuración

Para las opciones del archivo de configuración, hay una pequeña explicación en [la página de NCRFpp](https://github.com/jiesutd/NCRFpp/blob/master/readme/Configuration.md), que no siempre es muy clara, por lo que voy a detenerme en las más importantes:

  * train_dir, dev_dir, test_dir, model_dir, word_emb_dir: a pesar del sufijo "dir" son, en realidad, los paths a los archivos de entrenamiento, desarrollo, test, modelo y embeddings, respectivamente.
  * seg: para etiquetación morfosintáctica, el valor tiene que ser True, a pesar de que no es el valor por defecto en NCRFpp-master/demo.train.config. Para reconocimiento de entidades tiene que ser False
  * word_emb_dim y char_emb_dim: son las dimensiones de los embeddings de palabras y caracteres. Si usas los embeddings de FastText, el valor de word_emb_dim tiene que ser 300.
  * use_crf: si True, especifica que hay que usar un CRF para generar la salida, si es False, se usa Softmax.
  * use_char: especifica si se usa un bloque para el procesamiento de caracteres (True) o no (Falso). 
  * word_seq_feature y char_seq_feature: especifican el tipo de red neuronal para el procesamiento de palabras y caracteres, respectivamente. Los valores admitidos en ambos parámetros son LSTM, GRU o CNN. En el paper 'Design Challenges and Misconceptions in Neural Sequence Labeling.pdf' sólo mencionan LSTM y CNN, y sólo hacen pruebas para esos dos tipos de red, lo que no carece de sentido, dado que tanto las LSTM como las GRU son redes recurrentes, y las LSTM suelen dar mejores resultados.
  * status: sirve para decirle al script si se va a entrenar un modelo ("train") o etiquetar (decode). En nuestro caso, es el valor que vamos a usar es train.
  * iteration: el número de iteraciones (epochs) para el entrenamiento.
  * batch_size: el tamaño del batch. El valor por defecto es 10, pero yo he usado 32, y no he tenido problemas.
  * ave_batch_loss: aunque el valor por defecto es False, yo he usado True, siguiendo el consejo [de los autores de la herramienta](https://github.com/jiesutd/NCRFpp/blob/master/readme/hyperparameter_tuning.md), donde se dice que True generalmente hace converger más rápido, y, a veces, con mejores resultados.
  * learning_rate: he dejado el valor por defecto de 0.015, que es el que aconsejan [en la página de GitHub de NCRF++](https://github.com/jiesutd/NCRFpp/blob/master/readme/hyperparameter_tuning.md) para reconocimiento de entidades.

En el resto de parámetros he dejado los valores por defecto en que podeis encontrar en NCRFpp-master/demo.train.config. En general, son los mismos que usaron en los experimentos del paper 'Design Challenges and Misconceptions in Neural Sequence Labeling'.

**Importante**: Mirando en el archivo `NCRFpp-master/utils/data.py` he encontrado que hay algunos hiperparámetros del modelo que no aparecen en los archivos de configuración, y que, si se incluyen, dan un warning de que se ha redefinido el parámetro. Uno de ellos es `iteration`, que define el número máximo de epochs (iteraciones) en el entrenamiento. Por defecto, es 50. Pero si se incluye la línea:

```
iteration=[num]
```

al final del fichero de configuración, en la sección donde se definen los hiperparámetros (`###Hyperparameters###`), se actualiza el número máximo de iteraciones a `[num]`, siempre y cuando sea un entero positivo.

## Resultados

Al igual que NeuroNLP2, `main.py` escribe a la consola, por lo que hay que redireccionar la salida de la misma a un archivo. En cada iteración NCRF++ escribe los resultados obtenidos sobre el conjunto de desarrollo (dev) y sobre el conjunto de prueba (test). Si estamos haciendo una tarea de segementación, como reconocimiento de entidades nombradas, (el parámetro `seg` del archivo de configuración tiene valor `True`), las líneas relevantes de la salida tendrán la siguiente forma:

```
Dev: time: ..., speed: ...; acc: [número], p: [número], r: [número], f: [número]
...
Test: time: ..., speed: ...; acc: [número], p: [número], r: [número], f: [número]
```

Si estamos en una tarea de en donde el parámetro `seg` tiene valor `False`, como etiquetación morfosintáctica, las líneas relevantes de la salida tendrán la siguiente forma:

```
Dev: time: ..., speed: ...; acc: [número]
...
Test: time: ..., speed: ...; acc: [número]
```
