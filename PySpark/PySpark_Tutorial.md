# Curso guiado de PySpark — Notebook mejorado

> Versión mejorada del notebook que me enviaste. Organizado en secciones claras, con mejores explicaciones en Markdown, manejo de errores, buenas prácticas y código más robusto para ejecutar en Jupyter (Windows / Linux / macOS).

---

## Objetivos

- Configurar y comprobar PySpark en tu entorno local
- Crear `SparkSession` con configuración razonable para notebooks
- Cargar datos (DataFrame desde lista, leer CSV) y hacer transformaciones comunes (select, filter, agregaciones, joins)
- Guardar/leer Parquet y entender particionado
- Buenas prácticas: caché, broadcast joins, toPandas, manejo de memoria
- Siguientes pasos sugeridos (performance, streaming, MLlib)

---

## Recomendaciones previas (rápido)

1. PySpark requiere Java (OpenJDK). Recomendado: JDK 17 o 11.
2. En Windows, evita usar `quit()` dentro del notebook porque cerrará el kernel. Usa instrucciones de shell con `!` o `subprocess` para instalar.
3. Siempre usa `sys.executable -m pip install ...` para instalar paquetes desde un notebook para apuntar al Python correcto.

> **Nota:** Si trabajas con Anaconda / conda, instala `openjdk` y `pyspark` desde conda si prefieres.

---

## 1) Comprobaciones de entorno (célula)

```python
# Cell: comprobaciones de entorno
import sys, shutil, subprocess

print("Python:", sys.version.splitlines()[0])

java = shutil.which("java")
if java:
    try:
        out = subprocess.check_output([java, "-version"], stderr=subprocess.STDOUT)
        print(out.decode(errors='ignore').splitlines()[0])
    except Exception as e:
        print("Error ejecutando 'java -version':", e)
else:
    print("No se encontró 'java' en PATH. PySpark necesita Java (OpenJDK 11/17 recomendado).")

try:
    import pyspark
    print("pyspark:", pyspark.__version__)
except Exception:
    print("pyspark no instalado. Ejecuta la celda de instalación si quieres instalarlo.")
```

---

## 2) Instalación (ejecutar solo si hace falta)

> Ejecuta esto SOLO si `pyspark` no está instalado. Ajusta versiones si lo deseas.

```python
# Cell: instalar paquetes (Windows/Linux/macOS)
import sys

print("Usando Python:", sys.executable)
!"{sys.executable}" -m pip install --upgrade pip
!"{sys.executable}" -m pip install pyspark==3.5.1 pandas matplotlib --quiet
print("Instalación finalizada. Reinicia el kernel si se indica.")
```

**Consejo**: en Windows, si instalas el JDK, añade la carpeta `bin` de JAVA_HOME al PATH y reinicia el terminal / kernel.

---

## 3) Crear SparkSession (segura para notebooks)

```python
# Cell: crear SparkSession con manejo
from pyspark.sql import SparkSession
from pyspark import SparkConf

conf = SparkConf()
conf.set("spark.sql.shuffle.partitions", "8")  # ajusta a tu CPU
conf.set("spark.driver.memory", "2g")

try:
    spark = SparkSession.builder.appName("PySpark-Interactive-Notebook").config(conf=conf).getOrCreate()
    print("SparkSession creada. master=", spark.sparkContext.master, "version=", spark.version)
except Exception as e:
    print("Error creando SparkSession:", e)

# Si necesitas parar al final del notebook:
# spark.stop()
```

**Por qué:** establecer `shuffle.partitions` y `driver.memory` en el notebook hace que los trabajos pequeños no generen demasiados fragmentos.

---

## 4) DataFrame desde lista — ejemplo y operaciones básicas

```python
# Cell: ejemplo pequeño
from pyspark.sql import functions as F

data = [
    ("Alice", "2025-01-01", 50.0),
    ("Bob",   "2025-01-01", 20.0),
    ("Alice", "2025-01-02", 30.0),
    ("Bob",   "2025-01-02", 40.0),
    ("Carmen","2025-01-02", 15.0)
]

columns = ["name", "date", "amount"]
# Puedes forzar tipos con schema si lo deseas (pyspark.sql.types)
df = spark.createDataFrame(data, schema=columns)
df.show()
df.printSchema()

# Filtrar
print('\nFiltrando amount > 25:')
df.filter(F.col('amount') > 25).show()

# Agrupar y agregar
agg = df.groupBy('name').agg(
    F.count('*').alias('n_trx'),
    F.sum('amount').alias('total_amount'),
    F.avg('amount').alias('avg_amount')
).orderBy(F.desc('total_amount'))
agg.show()
```

---

## 5) Leer/Escribir CSV y Parquet (ejemplo reproducible)

```python
# Cell: crear CSV de ejemplo y leerlo con Spark
import pandas as pd
sample_csv = 'sample_data.csv'
pd.DataFrame([
    {'name':'Alice','date':'2025-01-01','amount':50},
    {'name':'Bob','date':'2025-01-01','amount':20},
    {'name':'Carmen','date':'2025-01-03','amount':70},
]).to_csv(sample_csv, index=False)

# Leer CSV con Spark (más seguro: especificar schema en producción)
df_csv = spark.read.option('header', True).option('inferSchema', True).csv(sample_csv)
df_csv.show()

df_csv.printSchema()
print('Count:', df_csv.count())

df_csv.describe('amount').show()

# Escribir Parquet particionado por 'date'
out_dir = 'out_parquet_example'
df_csv.write.mode('overwrite').partitionBy('date').parquet(out_dir)
print('Escrito parquet en:', out_dir)

# Leer de nuevo
from pathlib import Path
if Path(out_dir).exists():
    df_parq = spark.read.parquet(out_dir)
    df_parq.show()
else:
    print('Directorio no encontrado:', out_dir)
```

**Explicación rápida del particionado:** particionar por columnas con bajas cardinalidades (ej. `year`, `date`) mejora lecturas filtradas, pero no abuses con columnas de alta cardinalidad.

---

## 6) Joins y broadcast join para tablas desiguales

```python
# Cell: joins
customers = spark.createDataFrame([('Alice','A1'),('Bob','B1'),('Carmen','C1')], ['name','cust_id'])
transactions = df  # ejemplo anterior

# Join normal
joined = transactions.join(customers, on='name', how='left')
joined.show()

# Si customers es pequeño, usar broadcast para evitar shuffle
from pyspark.sql.functions import broadcast
joined_b = transactions.join(broadcast(customers), on='name', how='left')
joined_b.show()
```

---

## 7) Caché / persistencia y explain

```python
# Cell: caché y explain
agg.cache()  # o agg.persist(StorageLevel.MEMORY_ONLY)
print('Plan físico:')
agg.explain(True)

# Después de usarlo
agg.unpersist()
```

**Consejo:** cachea solo cuando vas a reusar un DataFrame varias veces y si hay memoria libre.

---

## 8) toPandas y visualización (advertencia de memoria)

```python
# Cell: usar toPandas solo con datasets pequeños
pdf = agg.toPandas()
print(pdf)

import matplotlib.pyplot as plt
pdf.plot.bar(x='name', y='total_amount', legend=False)
plt.title('Total amount por cliente')
plt.tight_layout()
plt.show()
```

> `toPandas()` trae todo a memoria local; evita en datasets grandes.

---

## 9) Buenas prácticas y pasos siguientes

- Evita `inferSchema` en producción: define `StructType` si necesitas control de tipos.
- Para I/O masiva usa rutas `s3://` o `hdfs://` y habilita credenciales apropiadas.
- Usa `explain()` y la Spark UI (por defecto en `http://localhost:4040` mientras el driver esté activo) para entender shuffles.
- Para streaming: mira `Structured Streaming` (readStream / writeStream).
- Para ML: `pyspark.ml` ofrece pipelines y transformers; usa `VectorAssembler` y `Pipeline`.

---

## 10) Plantilla de finalización (parar Spark)

```python
# Cell: al terminar
spark.stop()
print('Spark detenido')
```

---

## Problemas comunes y soluciones rápidas

- **No se encuentra java**: instala JDK, establece `JAVA_HOME` y añade `bin` al `PATH`.
- **Permisos con setx en Windows**: ejecuta PowerShell/Command Prompt como Administrador.
- **`pyspark` import error**: revisa que `pip` instaló en el mismo Python que usa el kernel.

---

Si quieres, puedo convertir este documento en un **.ipynb** real que puedas descargar y abrir en Jupyter, o bien adaptarlo a tu dataset concreto (ponme la ruta o describe la estructura de los ficheros y preparo las celdas de ETL/transformación).

