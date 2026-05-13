# Análisis Descriptivo y Muestreo Estadístico de Imágenes Astronómicas

## Descripción General

Este proyecto implementa un pipeline completo para el análisis descriptivo de imágenes astronómicas y la generación de muestras estadísticamente significativas. Utiliza técnicas de estadística inferencial para garantizar que la muestra seleccionada sea representativa de la población total, permitiendo posteriores estudios sobre un conjunto de datos validado.

### Características Clave

- **Análisis Descriptivo Automático**: Extrae características visuales de imagen (intensidad, entropía, sharpness)
- **Muestreo Estadístico Riguroso**: Uso de la fórmula de Cochran para determinar tamaño de muestra significativo
- **Muestreo Estratificado**: Agrupa imágenes con KMeans y muestrea proporcionalmente de cada grupo
- **Validación Estadística**: Pruebas de Kolmogorov-Smirnov para validar la representatividad de la muestra

---

## Estructura del Proyecto

```
proyectoFinalAA/
├── src.ipynb                          # Notebook de exploración inicial
├── data_pipeline.py                   # Pipeline principal de procesamiento
├── analyze_images.py                  # Script de análisis exploratorio
├── pyproject.toml                     # Configuración de dependencias
├── data/                              # Directorio con imágenes (.jpg)
```

---

## Cómo Usar

### 1. Configuración Inicial

Asegúrate de tener Python 3.14+ e instala `uv` si no lo tienes:

```bash
# Instalar uv (si no lo tienes)
pip install uv

# Instalar dependencias con uv
uv sync
```

Para más información sobre `uv`, consulta su [documentación oficial](https://docs.astral.sh/uv/).

### 2. Preparar los Datos

Coloca tus imágenes astronómicas en formato `.jpg` en el directorio `data/`:

```
data/
├── imagen_001.jpg
├── imagen_002.jpg
├── imagen_003.jpg
└── ...
```

### 3. Ejecutar el Pipeline Completo

#### **Opción A: Ejecutar mediante Script Python**

```bash
python data_pipeline.py
```

Este comando ejecuta el flujo completo:
1. Extrae features de todas las imágenes
2. Calcula el tamaño de muestra estadísticamente significativo
3. Realiza muestreo estratificado
4. Valida que la muestra sea representativa
5. Genera reportes y visualizaciones

#### **Opción B: Exploración Interactiva (Notebook)**

```bash
# Abre el notebook en Jupyter
jupyter notebook src.ipynb
```

Este notebook contiene:
- Análisis exploratorio del dataset
- Visualizaciones de características
- Código educativo paso a paso
- Experimentación interactiva

#### **Opción C: Análisis Exploratorio Rápido**

```bash
python analyze_images.py
```

Este script genera un análisis exploratorio con:
- Estadísticas descriptivas del dataset completo
- Histogramas de distribuciones de características
- Visualizaciones de agrupamientos por características
- Estadísticas detalladas por canal RGB y otras métricas

---

## Flujo del Pipeline (`data_pipeline.py`)

### Paso 1: Extracción de Features (`extract_features()`)

Procesa cada imagen y extrae:
- **Metadatos**: Dimensiones, formato
- **Estadísticas de Intensidad**: Media, desviación estándar, mín, máx
- **Entropía de Shannon**: Medida de complejidad visual
- **Sharpness**: Nitidez basada en la varianza del Laplaciano

**Output**: `full_dataset_stats.csv`

### Paso 2: Cálculo de Tamaño de Muestra (`calculate_sample_size()`)

Utiliza la **Fórmula de Cochran** para determinar un tamaño de muestra estadísticamente significativo:

$$n = \frac{n_0}{1 + \frac{n_0 - 1}{N}}$$

Donde:
- $n_0 = \frac{Z^2 \cdot p \cdot (1-p)}{E^2}$
- $Z$ = Valor crítico (1.96 para 95% confianza)
- $p$ = Proporción esperada (0.5 por defecto)
- $E$ = Margen de error (3% por defecto)
- $N$ = Tamaño total de población

**Por defecto**: 95% confianza, 3% margen de error

### Paso 3: Muestreo Estratificado (`perform_sampling()`)

1. Normaliza las características usando StandardScaler
2. Agrupa imágenes en 5 estratos usando KMeans (basado en similitud de características)
3. Selecciona muestras **proporcionalmente** de cada estrato
4. Garantiza representatividad equilibrada en el espacio de características

**Output**: `significant_sample.csv`

### Paso 4: Validación Estadística (`validate_sample()`)

Ejecuta la **Prueba de Kolmogorov-Smirnov** para:
- Comparar distribuciones de población vs muestra
- Verificar que no hay diferencias significativas (p-value > 0.05)
- Generar visualizaciones de comparación

**Output**: `sampling_validation.png`

---

## Outputs Generados

| Archivo | Descripción |
|---------|-------------|
| `full_dataset_stats.csv` | Features de todas las imágenes |
| `significant_sample.csv` | Muestra representativa del dataset |
| `corrupted_files.csv` | Imágenes no procesables (si las hay) |
| `sampling_validation.png` | Gráficas KS test y distribuciones |

---

## Personalización

### Ajustar Parámetros de Muestreo

Edita `data_pipeline.py` en la función `run_workflow()`:

```python
# Aumentar confianza (99%) o reducir margen de error
n_target = calculate_sample_size(
    N, 
    confidence_level=0.99,   # 0.90, 0.95, 0.99
    margin_error=0.02        # Reduce para muestra más grande
)
```

### Cambiar Número de Clusters

En la función `perform_sampling()`:

```python
n_clusters = 5  # Aumenta para más granularidad
```

---

## Dependencias

```
ipykernel>=7.2.0          # Para Jupyter notebooks
matplotlib>=3.10.9         # Visualizaciones
numpy>=2.4.4               # Computación numérica
pandas>=3.0.3              # Manipulación de datos
pillow>=12.2.0             # Procesamiento de imágenes
scikit-image>=0.26.0       # Análisis de imágenes avanzado
scikit-learn>=1.8.0        # Machine learning (KMeans, escalado)
tqdm>=4.67.3               # Barras de progreso
scipy                      # Estadística (Laplaciano, KS test)
```

---

## Conceptos Estadísticos Utilizados

### Fórmula de Cochran
Calcula el tamaño de muestra necesario para garantizar representatividad estadística de una población.

### Muestreo Estratificado
Divide la población en subgrupos homogéneos (estratos) y muestrea proporcionalmente de cada uno, mejorando la representatividad.

### Prueba de Kolmogorov-Smirnov
Contrasta si dos distribuciones de probabilidad son diferentes. Usamos p-value > 0.05 como criterio de "no significancia".

### Entropía de Shannon
Medida de la cantidad de información/complejidad en una imagen.

### Sharpness (Laplaciano)
Medida de la nitidez de una imagen basada en la varianza del filtro Laplaciano.

---

## Validación del Pipeline

Para verificar que todo funciona correctamente:

```bash
# Verifica que el dataset tenga al menos 100 imágenes
ls data/ | wc -l

# Ejecuta el pipeline
python data_pipeline.py

# Verifica los outputs
ls -lh *.csv *.png
```

**Esperado**: 
- ✓ `full_dataset_stats.csv` generado
- ✓ `significant_sample.csv` generado
- ✓ `sampling_validation.png` generado
- ✓ Todos los p-values > 0.05

---

## Uso de Resultados

Una vez generada la muestra significativa:

1. **Visualización Interactiva**: Explora los resultados usando `src.ipynb` para análisis detallados
2. **Análisis Posterior**: Utiliza `significant_sample.csv` para estudios específicos o modelado adicional
3. **Validación de Hipótesis**: Confirma que la muestra es representativa mediante las pruebas KS generadas
4. **Documentación**: Los archivos CSV generados contienen toda la información necesaria para reportes y publicaciones

---

## Autor

Proyecto Final - Análisis Automático de Imágenes Astronómicas  
**UNAM | 2026**

---

## Notas

- Las imágenes deben estar en formato `.jpg`
- El script es tolerante a errores
- Todos los archivos CSV usan UTF-8 encoding
- Las visualizaciones se generan en 300 DPI para publicación

