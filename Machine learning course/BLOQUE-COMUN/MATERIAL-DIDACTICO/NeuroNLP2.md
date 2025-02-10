# NeuroNLP2

## Descripción

NeuroNLP2 es una herramienta que implementa varios modelos de redes de neuronas para tareas de Procesamiento del Lenguaje Natural: etiquetación morfosintáctica, reconocimiento de entidades nombradas y análisis sintáctico de dependencias. Para las dos primeras, implementa la siguiente arquitectura:

- Una capa CNN (*Convolutional Neural Network*) para extraer una representación a nivel de caracteres del texto de entrada, tanto para entrenamiento como para test. Esa representación a nivel de carácter se concatena con las representaciones vectoriales (*word embeddings*) de cada palabra. Los *embeddings* son aportados por el usuario.
- Uno o más elementos LSTM (*Long Short-Term Memory*), un tipo de red recurrente que se usa para procesar secuencias de entrada cuya longitud no se conoce de antemano y permite construir una representación en forma de vector para cada ejemplo (en nuestro caso palabras) usando no sólo la información de dicho ejemplo, sino también los ejemplos anteriores. NeuroNLP2 usa elementos *BiLSTM*, que consisten en dos capas LSTM: una que lee la entrada de izquierda a derecha, y otra que lee la entrada de derecha a izquierda
- Una capa opcional que implementa un modelo de campo condicional (*Conditional Random Fields* o CRF). Los CRFs son modelos para etiquetación de secuencias, que, en lugar de asignar una etiqueta a cada ejemplo por separado, intentan encontrar la secuencia de etiquetas más probable para una secuencia de ejemplos. En nuestro caso, el CRF, con la información procedente del LSTM, intenta encontrar la secuencia de etiquetas más probable para la palabras de la frase de entrada, en lugar de asignar la etiqueta más probable a cada palabra por separado.

En el caso de análisis sintáctico, la última capa, en lugar de ser un modelo de campo condicional, implementa un algoritmo de análisis sintáctico de dependencias: MST, Stack Pointer o Biaffine.

La arquitectura de NeuroNLP2 se describe en los siguientes papers:

- [Este paper](http://www.cs.cmu.edu/~xuezhem/publications/P16-1101.pdf) describe la arquitectura seguida en tareas de etiquetación de secuencias, como etiquetación morfosintáctica o reconocimiento de entidades nombradas.
- Esa arquitectura básica es usada para implementar 3 tipos de analizadores sintácticos de dependencias: [MST](https://www.aclweb.org/anthology/I17-1007/), [Stack Pointer](https://arxiv.org/pdf/1805.01087.pdf) y Biaffine.

No es necesario leer los papers y entender como funcionan los diferentes analizadores para usar la herramienta, pero os dejo los enlaces por si queréis leerlos.
  
NeuroNLP2 se puede descargar desde [GitHub](https://github.com/XuezheMax/NeuroNLP2)

## Instalación

Requerimientos: versión 3.6 o superior de Python, PyTorch y Gensim. Con `pip` se pueden instalar con los siguientes comandos:

```
pip3 install overrides   (este lo necesité en alguna instalación, pero es posible que no os haga falta)

pip3 install fasttext
pip3 install gensim

pip3 install torch torchvision
```

Para hacer experimentos, os he abierto una cuenta en idefix (193.147.87.32), un PC ubicado en nuestro laboratorio y que tiene dos GPUs. La cuenta tiene como login "visitante" y como password "sivispacem". He creado dos directorios en dicha cuenta:

- "Corpora" contiene corpora etiquetados descargados de [Universal Dependencies](https://universaldependencies.org/#download)
- "Taggers" contiene un directorio con la última versión de NeuroNLP2 (Taggers/NeuroNLP2)
En principio, deberíais poder conectaros a idefix a través de cualquier terminal que use ssh, o de un cliente de RDC, aunque a veces estos últimos no funcionan bien desde fuera de la red de la universidad. También podéis copiar archivos con cualquier herramienta que soporte los protocolos sftp o scp.

## Estructura del código

El código y archivos de configuración de NeuroNLP2 están distribuidos en el siguiente árbol de directorios:

```
NeuroNLP2
├── experiments
│   ├── configs
│   ├── data
│   ├── eval
│   ├── logs
│   ├── models
│   └── scrips
├── LICENSE
├── neuronlp2
│   ├── io
│   ├── models
│   ├── nn
│   ├── optim
│   └── tasks
└── README.md
```

En `neuronlp2/` se encuentran las implementaciones de los modelos usados para etiquetación (`models/sequence_labeling.py`) y para análisis sintáctico de dependencias (`models/parsing.py`) e implementaciones de capas de red propias de la herramienta (`models/nn/modules.py`). En principio, no es necesario trabajar con esta parte del código.

En la carpeta `experiments/` es donde está el código, archivos de configuración y datos para las diferentes utilidades de Procesamiento del Lenguaje Natural implementadas en la herramienta. En nuestro caso particular, el código que implementa el etiquetador morfosintáctico está en el script `experiments/pos_tagging.py`, los analizadores sintácticos están en `experiments/parsing.py`, y el código para reconocimiento de entidades está en `experiments/ner.py`. Los directorios `experiments/data` y `experiments/models` aparecen referenciados en scripts de ejecución contenidos directorio `scripts`, pero que no están creados en el árbol de directorios que podemos bajar de GitHub. Por lo tanto, habrá que crearlos.

De manera más detallada:

```
experiments
├── configs           archivos JSON con parámetros a usar en distintos modelos
│   ├── ner
│   │   └── conll03.json
│   ├── parsing
│   │   ├── biaffine.json
│   │   ├── neuromst.json
│   │   └── stackptr.json
│   └── pos
│       └── wsj.json
│       └── anCora.json
├── data             (aparece en los scripts de ejecución, por lo que hay que crearlo)
├── eval
│   ├── conll03eval.v2
│   └── conll06eval.pl
├── logs
├── models           (aparece en los scripts de ejecución, por lo que hay que crearlo)
├── ner.py           script para tareas de reconocimiento de entidades
├── parsing.py       script para tareas de parsing
├── pos_tagging.py   script para tareas de etiquetación morfosintáctica
├── scripts          scripts shell para lanzar las ejecuciones
│   ├── run_analyze.sh      (no)     
│   ├── run_deepbiaf.sh     (análisis sintactico Biaffine)       
│   ├── run_ner_conll03.sh  (reconocimiento de entidades nombradas)
│   ├── run_neuromst.sh     (análisis sintactico MST)
│   ├── run_pos_wsj.sh      (etiquetación morfosintáctica)
│   └── run_stackptr.sh     (análisis sintactico Stack pointer)
└── shuffle_conllu.py  script Python para aleatorizar y "cortar" los ficheros conll
```

En el código que os hemos dejado en idefix (193.147.87.32) hemos introducido una modificación en los scripts `ner.py` y `pos_tagging.py` consistente en usar el mismo tamaño de batch durante el entrenamiento de la red y la evaluación de resultados. Anteriormente, se usaba un tamaño de batch de 256 en la evaluación, que a veces daba lugar a que se agotara la memoria de la GPU durante dicha evaluación. Si os encontráis con que, al hacer un experimento, el script se interrumpe, quejándose de que no puede reservar suficiente memoria en la GPU, reducid el tamaño del batch (lo que se puede hacer en las opciones de línea de comandos de ambos scripts). Recordad que, aunque no es obligatorio, el tamaño de batch suele ser una potencia de 2, para aprovechar mejor los núcleos de la GPU (cuyo número suele ser también una potencia de 2).

La carpeta `experiments/scripts` contiene scripts de `bash` para la ejecución de las diferentes herramientas implementadas en los scripts python del directorio `experiments/`. Por ejemplo, el script `run_pos_wsj.sh` ejecuta `pos_tagging.py` usando como datos de entrenamiento el Corpus del Wall Street Journal (WSJ) que forma parte del Penn Treebank. Este es un corpus muy usado para la evaluación de etiquetadores en inglés. Por su parte, los scripts `run_deepbiaf.sh`, `run_neuromst.sh` y `run_stackptr.sh` ejecutan los tres tipos de analizadores sintácticos implementados en NeuroNLP2.

**Importante:** en los scripts anteriores se usa el comando 'python' para ejecutar las diferentes utilidades de la herramienta (`pos_tagging.py`/`parsing.py`/`ner.py`). En idefix, la versión por defecto de python es python 2.7, pero NeuroNLP2 está implementado en la versión 3 de python. Por lo tanto, hay que, o bien cambiar el comando en run_pos_wsj.sh a "python3" o incluir un alias en el archivo .bashrc de la cuenta en idefix, de modo que el comando `python` llame a python3.

El código en `run_pos_wsj.sh` es el siguiente:

```
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python pos_tagging.py --config configs/pos/wsj.json --num_epochs 400 --batch_size 32 \  
 --loss_type sentence --optim sgd --learning_rate 0.01 --lr_decay 0.99999 --grad_clip 0.0 --warmup_steps 10 --weight_decay 0.0 --unk_replace 0.0 \  
 --embedding glove --embedding_dict "data/glove/glove.6B/glove.6B.100d.gz" --model_path "models/pos/wsj" \  
 --train "data/POS-penn/wsj/split1/wsj1.train.original" --dev "data/POS-penn/wsj/split1/wsj1.dev.original" --test "data/POS-penn/wsj/split1/wsj1.test.original" 
```

Vamos a detenernos en algunas de las opciones de línea de comandos de pos_tagging.py:

`--config` carga un fichero de configuración (en el ejemplo configs/pos/wsj.json), proporcionado por el usuario, en el que se pueden modificar varios de los parámetros de la red de neuronas (desconozco por qué los autores de NeuroNLP2 han dispuesto que algunos hiperparámetros de la red se carguen en un archivo JSON y otros como opciones en línea de comandos del script). En particular, el parámetro `embed_dim` especifica el tamaño de los embeddings. Está puesto a 100 en configs/pos/wsj.json, pero puede que tengáis que modificarlo según los embeddings usados (ver más abajo).

`--num_epochs` es el número máximo de iteraciones (epochs) que se llevarán a cabo durante el entrenamiento. 400 o 1000 son los valores que suelen aparecen en los scripts del directorio `scripts`, pero yo comenzaría con un valor más pequeño, por ejemplo 100. Al entrenar el modelo en cada epoch, `pos_tagging.py`/`parsing.py`/`ner.py` van calculando el error del mismo en el conjunto de datos de desarrollo (dev) y, cuando obtiene un error menor que en las epoch anteriores, almacena el modelo actual. Al final, el modelo generado es el que tiene menor error en el conjunto de desarrollo.

`--batch_size` determina el tamaño del batch (en nuestro caso el número de frases) usado en cada ciclo de entrenamiento o test. Ya lo he mencionado más arriba.

`--embedding` determina el tipo de embedding, con 4 posibilidades (glove, sskip, senna y polyglot). En la practica, como los embeddings son todos vectores de números reales, el código usará cualquier embedding de la misma manera. El especificar el tipo de embedding sirve para decirle a pos_tagging.py cual es el formato del archivo de embeddings que puede esperar. Si queréis saber como son dichos formatos, podéis echar un vistazo en la función `load_embedding_dict()` del script `neuronlp2/utils.py`, que es la encargada de cargar los embeddings desde un archivo.

`--embedding_dict` sirve para especificar el path al archivo con los embeddings. Haciendo experimentos, he guardado los archivos de embeddings en el directorio `experiments/data/`. Dado que vamos a necesitar embeddings para múltiples lenguajes, una buena opción son los proporcionados por FastText, que tiene embeddings para 157 lenguas (ver más abajo). Para cargarlos, hay que especificar `sskip` como tipo de embedding, en la opción `--embedding`.

NeuroNLP2 espera embeddings de texto comprimidos, por lo que, en caso de usar los ficheros de FastText, hay que descargar los archivos de los enlaces "text" para cada lenguaje que se use, y **no** descomprimirlos.
  
Los embeddings de FastText son de dimensión 300, por lo que habrá que reflejar ese dato en la opción `embed_dim` del archivo de configuración JSON que corresponda.
  
`--model_path` es el path del directorio en donde se escribirá el modelo resultante del entrenamiento. Para evitar machacar la información de diferentes experimentos, lo mejor es crear un directorio para cada uno. Por ejemplo: `experiments/models/pos/[corpus]/`.

`--train`, `--dev`, y `--test` especifican los paths a los archivos con los conjuntos de datos de (respectivamente) entrenamiento (train), desarrollo (dev) y prueba (test). El primero se usa para entrenar el etiquetador, y los otros dos para calcular la precisión del mismo (el porcentaje de etiquetas correctas generadas por el etiquetador cuando recibe como entrada dichos conjuntos).

El formato de los archivos con los datos de entrenamiento, validación y prueba es el usado en la competición CoNLL-X para la etiquetación morfosintáctica y análisis sintáctico, y se especifica en la [página de NeuroNLP2](https://github.com/XuezheMax/NeuroNLP2/issues/9) y, más detalladamente, en la página 3 de este [documento](https://ilk.uvt.nl/~emarsi/download/pubs/14964.pdf). Es un formato para una tarea de análisis sintáctico de dependencias, con 10 columnas separadas por tabuladores, y una línea para cada palabra (dejando una línea en blanco entre una frase y la siguiente). En cualquier tarea que no sea de análisis sintáctico se pueden dejar en blanco algunas de las columnas (lo que se hace con el carácter `_`) excepto las columnas 0 (índice de la palabra en la frase), 1 (la propia palabra), 4 (la etiqueta morfosintáctica) y 6 (la posición en la frase de la cabeza de la palabra actual en una relación de dependencia). Esta última columna es información usada en el análisis sintáctico, pero, en el caso de etiquetación morfosintáctica, `pos_tagging.py` pide que haya un número en dicha columna (probablemente para poder usar el mismo formato, y el mismo código para leerlo, en los scripts de etiquetación y análisis sintáctico).

En el caso del reconocimiento de entidades, tenemos un formato con 5 columnas, separadas por espacios: la posición de la palabra en la frase (comenzando por 1), la propia palabra, la etiqueta morfosintáctica, una etiqueta especificando a qué sintagma pertenece y, finalmente, una etiqueta especificando si pertenece o no a una entidad nombrada. Tenemos un ejemplo [en la misma página que el formato CoNNL-X](https://github.com/XuezheMax/NeuroNLP2/issues/9).


El rendimiento de los modelos basados en redes de neuronas es, a veces, muy dependiente de la inicialización del modelo. En el caso particular de NeuroNLP2 a veces me he encontrado con que el etiquetador obtenía resultados excelentes desde el primer epoch (97% o más de precisión en experimentos con WSJ), mientras que, en otros experimentos con los mismos hiperparámetros y datos de entrada, los resultados eran de pena (10% de precisión durante la mayor parte del entrenamiento). Es posible que merezca la pena hacer varios intentos hasta ver si se pueden obtener buenos resultados.

## Archivos de configuración

Para etiquetación morfosintáctica o reconocimiento de entidades nombradas. Por ejemplo: `configs/parsing/wsj.json'

```
{
  "crf": true,               //Si true, usar un CRF para generar la salida
  "bigram": true,
  "embedd_dim": 100,         //Dimensión de los embeddings. Para FastText tiene que ser 300
  "char_dim": 30,            //Dimensión de los embeddings de caracteres
  "rnn_mode": "LSTM",        //Tipo de red recurrente usada: RNN, LSTM, FastLSTM, GRU
  "num_layers":1,            //Número de capas RNN
  "hidden_size": 256,        //Número de elementos en las capas RNN
  "out_features": 256,       //Dimensión de los vectores de cada elemento de salida
  "dropout": "std",          //Tipo de Dropout: std o variational
  "p_in": 0.33,              //%Dropout
  "p_out": 0.5,              //%Dropout
  "p_rnn": [0.33, 0.5],      //%Dropout
  "activation": "elu"        //Función de activación: elu o tanh
}
```

Para análisis sintáctico. Por ejemplo: `configs/parsing/biaffine.json`

```
{
  "model": "DeepBiAffine",   //Tipo de analizador: DeepBiAffine, NeuroMST o StackPtr
  "word_dim": 100,           //Dimensión de los embeddings. Para FastText tiene que ser 300
  "char_dim": 100,           //Dimensión de los embeddings de caracteres
  "pos": true,               //Si True, se usan etiquetas morfosintácticas
  "pos_dim": 100,            //Dimensión de los embeddings de etiquetas morfosintácticas
  "rnn_mode": "FastLSTM",    //Tipo de red recurrente usada: RNN, LSTM, FastLSTM, GRU
  "num_layers":3,            //Número de capas RNN
  "hidden_size": 512,        //Número de elementos en las capas RNN
  "arc_space": 512,          //Número de elementos en la capa que realiza el análisis sintáctico
  "type_space": 128,         //Dimensión de los vectores de cada elemento de salida
  "p_in": 0.33,              //%Dropout
  "p_out": 0.33,             //%Dropout
  "p_rnn": [0.33, 0.33],     //%Dropout
  "activation": "elu"        //Función de activación: elu o tanh
}
```

## Word Embeddings

Descarga desde la página de [FastText](https://fasttext.cc/docs/en/crawl-vectors.html). Los enlaces directos están al final de la página. Sólo se usará la versión .vec.gz).

Son archivos muy grandes (~1,2GB comprimido >4GB descomprimido). **No hace falta descomprimirlos**.

## Corpora

Un recurso útil tanto para la etiquetación morfosintáctica como para el análisis sintáctico, son los Bancos de Árboles (*Treebanks*) de [Universal Dependencies](https://universaldependencies.org/)

- [Página de Descarga](https://universaldependencies.org/#download)

- [Ultima versión (2.7)](http://hdl.handle.net/11234/1-3424)

- [Enlace directo a corpus 2.7](https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3424/ud-treebanks-v2.7.tgz)

Al descomprimirlo (`tar xzvf ud-treebanks-v2.7.tgz`) en el directorio `ud-treebanks-v2.7` tenéis un subdirectorio para cada idioma.

La mayoría tienen una estructura similar. Por ejemplo, para el corpus AnCora en español, tenemos:

* `es_ancora-ud-train.conllu`    Corpus de entrenamiento en formato CoNLL-U  (usado durante el entrenamiento para entrenar los modelos)
* `es_ancora-ud-dev.conllu`      Corpus de validación en formato CoNLL-U     (usado durante el entrenamiento para validar y ajustar los parámetros de cada modelo) 
* `es_ancora-ud-test.conllu`     Corpus de test en formato CoNLL-U           (usado para comprobar el rendimiento final de los modelos)

También incluyen la versión sólo texto sin las anotaciones (`es_ancora-ud-dev.txt, es_ancora-ud-test.txt, es_ancora-ud-train.txt`)

El formato CONLLU se detalla [en la página de Universal Dependencies](https://universaldependencies.org/format.html).

El formato de los archivos de entrada que usa NeuroNLP2 es CoNLL-X, similar al formato CoNLL-U, pero no exactamente igual. Os hemos dejado un script para hacer la conversión (`conllu_to_conllx.pl`) en idefix (en `Taggers/NEuroNLP2/experiments/scripts/`). Está escrito en perl y obtiene la entrada desde la consola, volcando la salida a también a la consola. Por ejemplo:

```
perl ../../scripts/conllu_to_conllx.pl < es_ancora-ud-train.conllu > es_ancora-ud-train.conllx'
```

Para la tarea de reconocimiento de entidades nombradas, podeis encontrar los archivos de entrenamiento para la competición CoNLL-03 [aqui](https://github.com/glample/tagger/tree/master/dataset). Hay un archivo grande de entrenamiento y dos archivos más pequeños que se pueden usar para validación y prueba. Para poder usar esos archivos hay que pasarlos al formato aceptado por NeuroNLP2 para [reconocimiento de entidades](https://github.com/XuezheMax/NeuroNLP2/issues/9). Para ello, he dejado un script en `experiments/scripts/CoNLL2NeuroNLP2.py` para pasar del formato CoNLL-03 al aceptado por NeuroNLP2. Si ejecutamos `python3 CoNLL2NeuroNLP2.py -h`, tenemos las siguientes opciones:

```
USAGE: python CoNLL2NeuroNLP2.py [-h] -s sourceFile -d destFile
ARGUMENTS:
  -h:           print this help
  -s:           path to the file from CoNLL-03
  -d:           path to the NeuroNLP2-compatible file
```

Habría que crear un directorio para guardar los archivos de salida y transformar, uno a uno los archivos originales al formato compatible con la herramienta.


## Ejecución

Hemos dejado en idefix, en el directorio `Taggers/NeuroNLP2/experiments/scripts`, dos scripts de ejemplo para ejecutar NeuroNLP como etiquetador morfosintáctico y analizadores de dependencias (Biaffine), ambos sobre el corpus anCora (`run_pos_anCora.sh` y `run_deepbiaf_anCora.sh`, respectivamente).

Para ejecutar ambos ejemplos con éxito, tendríais que hacer lo siguiente:

* Descargar los embeddings para español de FastText (podéis cambiar los directorios si también lo hacéis en los scripts):

```
    cd Taggers/NeuroNLP2/data
    wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.es.300.vec.gz
```

* Descargar y ajustar corpus:

```
    wget https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3424/ud-treebanks-v2.7.tgz
    tar xzvf ud-treebanks-v2.7.tgz
    cp -rf ud-treebanks-v2.7/UD_Spanish-AnCora/ ~/Taggers/NeuroNLP2/data/
```

* Pasar los archivos de entrenamiento, validación y test a formato CoNLL-X

```
    cd ~/Taggers/NeuroNLP2/experiments/data/UD_Spanish-AnCora/
    perl ../../scripts/conllu_to_conllx.pl < es_ancora-ud-train.conllu > es_ancora-ud-train.conllx
    perl ../..//scripts/conllu_to_conllx.pl < es_ancora-ud-dev.conllu > es_ancora-ud-dev.conllx
    perl ../../scripts/conllu_to_conllx.pl < es_ancora-ud-test.conllu > es_ancora-ud-test.conllx
```

* Ejecutar el etiquetador o analizador sintáctico:

```
    cd ~/Taggers/NeuroNLP2/experiments/
    ./scripts/run_pos_anCora.sh
    ./scripts/run_deepbiaf_anCora.sh
```

Los scripts de NeuroNLP2 generan una buena cantidad de información y la vuelcan a la consola en forma de texto. En particular, en cada iteración generan los resultados sobre el conjunto de validación (dev) y, si dichos resultados mejoran los de las iteraciones anteriores, sobre el conjunto de prueba (test). Podéis almacenar esos resultados en un archivo de texto redireccionando la salida del script a dicho archivo. Por ejemplo:

```
./scripts/run_deepbiaf_anCora.sh > parsing/deepbiaf_anCora/results.txt 2>&1
```

o dentro del propio script `run_deepbiaf_anCora.sh`:

```
#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 python3 -u parsing.py --mode train --config configs/parsing/biaffine.json --num_epochs 100 --batch_size 32 \
...
--model_path "models/parsing/deepbiaf_anCora/" > models/parsing/deepbiaf_anCora/results.txt 2>&1
```

## Resultados

Para obtener los resultados tenéis que recorrer la salida hasta encontrar los resultados de la última iteración. A continuación, indico, para cada caso qué tenéis que buscar:

### Etiquetación morfosintáctica (`pos_tagging.py`)

Si queréis chequear los valores sobre el conjunto de validación (dev), en cada iteración:

```
    Dev  corr: ... acc: [número]
```

Para encontrar el valor final sobre el conjunto de prueba (test), buscar la última ocurrencia de:

```
    Best test corr: ... acc: [número]
```

### Análisis sintáctico (`parsing.py`)

Si queréis chequear los valores sobre el conjunto de validación (dev), en cada iteración:

```
    Evaluating dev:
    W. Punct: ... uas: [número], las: [número] ...
    Root: ... acc: [número]
```

Para encontrar el valor final sobre el conjunto de prueba (test), buscar la última ocurrencia de:

```
    best test W. Punct: ... uas: [número], las: [número] ...
    best test Root: ... acc: [número]
```

### Reconocimiento de entidades nombradas (`ner.py`)

Si queréis chequear los valores sobre el conjunto de validación (dev), en cada iteración:

```
    Dev  acc: [número], precision: [número], recall: [número], F1: [número]
```

Para encontrar el valor final sobre el conjunto de prueba (test), buscar la última ocurrencia de:

```
    Best test acc: [número], precision: [número], recall: [número], F1: [número]
```
