# Big Data Derby — Perfiles de carrera y detección de anomalías

Trabajo del curso de **Big Data** — Maestría en Ciencia de Datos, Universidad Pontificia Bolivariana (UPB).
Autor: **Sebastián Velasco Ardila**

Extiende el pipeline medallion del módulo `09-derby` (dataset [Big Data Derby 2022](https://www.kaggle.com/competitions/big-data-derby-2022), tracking GPS de carreras de caballos de NYRA) con un **segundo modelo de machine learning**, no supervisado, que agrupa a cada caballo-carrera según **cómo corrió** y detecta las carreras **atípicas**.

---

## Estructura

```
├── 00-setup/
│   └── config.py                          # constantes del módulo (+ T_GOLD_RACE_DYNAMICS, T_GOLD_RACE_PROFILES)
├── 03-gold/
│   └── 05_gold_race_dynamics.ipynb         # tabla Gold con features de forma del perfil de velocidad
└── 04-ml/
    └── 04_cluster_race_profiles.ipynb      # KMeans + anomalías + MLflow
```

**Orden de ejecución:** `05_gold_race_dynamics` → `04_cluster_race_profiles`.
Ambos notebooks dependen de las tablas Silver del módulo (`silver_derby_tracking`, `silver_derby_starts`) y de `config.py` (`%run ../00-setup/config`).

---

## Por qué un modelo distinto al clasificador top-3

| | Clasificador top-3 (existente) | Este modelo |
|---|---|---|
| Tipo | Supervisado (`GBTClassifier`) | No supervisado (`KMeans`) |
| Momento de uso | **Antes** de la largada | **Después** de la carrera |
| Objetivo | Predecir si el caballo termina top 3 | Describir **cómo** corrió el caballo |
| Tracking GPS | Prohibido (sería *data leakage*) | Permitido: es el fenómeno a describir |

El resultado de la carrera (`is_top3`, `position_at_finish`) **no entra al modelo**; se usa solo para interpretar los clusters.

---

## Parte 1 — `gold_derby_race_dynamics`

**Grano:** un caballo en una carrera (`track_id, race_date, race_number, program_number`). 14.900 filas.
**PK:** `start_id` = `sha2` de las claves concatenadas con `"||"`, idéntica a `gold_derby_ml_features` (14.899 de 14.900 cruzan; el restante no existe en `silver_derby_starts`).

### Features

| Feature | Qué captura |
|---|---|
| `n_frames`, `race_duration_s` | Tamaño del registro GPS (4 Hz) |
| `speed_q1` … `speed_q4` | Velocidad media en cada cuarto de la carrera (reparto por `floor` del avance relativo) |
| `early_speed_ratio` | `speed_q1` / promedio de los cuartos — fuerza del arranque |
| `late_speed_ratio` | `speed_q4` / promedio de los cuartos — capacidad de cierre |
| `time_to_top_speed_frac` | Fracción de la carrera en que se alcanza la velocidad máxima |
| `speed_cv` | Coeficiente de variación de la velocidad — regularidad del ritmo |
| `n_decel_events` | Caídas de más de 12 km/h entre frames consecutivos (0,25 s) |
| `max_decel_kmh` | Mayor caída de velocidad entre frames consecutivos |

### Decisiones de limpieza (justificadas con exploración de datos)

| Decisión | Valor | Justificación |
|---|---|---|
| Ruido GPS | `speed_kmh > 90` se descarta | Saltos de posición generaban velocidades de miles de km/h. Afecta al 0,03% de los frames, repartidos en 882 caballos: se eliminan frames, no caballos |
| Espera en el partidor | Se recortan los frames previos al arranque | 1.444 caballos tenían frames detenidos al inicio, que sesgaban `early_speed_ratio`. Arranque = primer frame con ≥ 5 km/h sostenido durante 1 s |
| Registro mínimo | `n_frames ≥ 100` | Excluye 15 registros cortados (≈0,1%); garantiza ≥ 25 frames por cuarto |
| Umbral de frenazo | −12 km/h (no −8) | Con −8 km/h el 32% de los caballos tendría "incidentes"; con −12 km/h, el 15,8% (percentil 0,1% de los cambios entre frames) |

---

## Parte 2 — Clustering y anomalías

**Pipeline:** `VectorAssembler` → `StandardScaler(withMean, withStd)` → `KMeans(k=4, seed=42)`
**Features:** `early_speed_ratio`, `late_speed_ratio`, `time_to_top_speed_frac`, `speed_cv`

### Elección de k

`k = 4`: codo del costo (de k = 4 a 5 la reducción cae de 8.439 a 1.620) con silhouette de 0,52. Aunque k = 3 tiene mayor silhouette (0,63), concentra el 73% de los caballos en un solo grupo que mezcla dos estrategias distintas.

### Estilos de carrera

| Estilo | % | `late_ratio` | Máx. velocidad en | % top 3 |
|---|---|---|---|---|
| Cerrador | 16% | 0,99 | 74% de la carrera | **46,9%** |
| Ritmo sostenido | 44% | 0,93 | 20% | 44,8% |
| Puntero que se apaga | 37% | 0,86 | 20% | 34,6% |
| Se desploma / detenido | 3% | 0,69 | 21% | 15,0% |

Lo que más separa a los estilos exitosos no es cuándo alcanzan su velocidad máxima, sino **cuánto ritmo conservan en el cierre**. Es una asociación descriptiva, no causal.

### Detección de anomalías

**Score:** distancia euclidiana al centroide del propio cluster (espacio estandarizado). **Umbral:** percentil 99 → 150 anomalías (1%).

La inspección manual, comparando cada anomalía con los demás caballos de su carrera, encontró cuatro causas distintas: **incidentes reales** (caballos que se detuvieron), **registros GPS cortados**, **picos de ruido GPS** y **efectos de la carrera** (registros que continúan después de la meta). El score sirve para priorizar la revisión, no como diagnóstico.

### Salidas

- **MLflow:** parámetros (incluidas las decisiones de limpieza de Gold), silhouette, costo, tamaño de clusters, curva completa del codo y modelo con firma.
- **Tabla `gold_derby_race_profiles`:** `start_id`, claves, `cluster`, `estilo`, `anomaly_score`, `is_anomaly` y `model_run_id` (linaje con el run de MLflow).

---

## Stack

Databricks (serverless, Unity Catalog) · PySpark · Spark MLlib · Delta Lake · MLflow
