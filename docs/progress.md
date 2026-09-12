# Estado del proyecto

## Feature en curso

La Feature 4b (escalera de comparación y segmentación supervisada) está cerrada en la rama
`feat/04b-supervised-segmentation`, con PR contra `main`. Siguiente según el orden de
CLAUDE.md: **Feature 5 — Elasticidad**, que traerá la curva de abandono de aceptación y el link
prediction.

## Feature 4b — Escalera de comparación y segmentación supervisada (cerrada)

### Por qué

La Feature 4 reportó que el grafo no le gana al baseline de atributos (R² fuera de muestra de
0,077 contra 0,011). Al medir el techo después quedó claro que la comparación corría muy por
debajo de él: **quedarse con el grupo de MCC del comercio, sin modelo, predice la carga
relativa fuera de muestra con R² 0,740**. El k-means no supervisado comprimía doce categorías
en ocho clústeres y tiraba justo lo que mueve el objetivo.

### Qué se hizo

- **Benchmark sin modelo** (`segment_by_mcc_group`): cada comercio se queda en su grupo de MCC.
  Es la referencia contra la que se lee toda la escalera.
- **Segmentación supervisada** (`supervised.py`): los segmentos son las hojas de un árbol de
  regresión sobre la carga relativa, ajustado solo con la mitad de entrenamiento, y cada
  segmento viene con la regla que lo define.
- **Una sola partición** (`evaluate.train_test_split`), compartida por todas las cifras fuera de
  muestra. La evaluación ordena por `merchant_id` porque la partición es posicional: un join
  que reordenara filas mediría al método supervisado sobre comercios que sí vio.
- **Información marginal** (`evaluate.marginal_information`): el mismo predictor sobre cada
  representación, sin el paso de clustering, para que ningún método se lleve el crédito de la
  compresión en vez del de la información.
- **Bloques de variables equilibrados** en el baseline.
- **Arreglo en `normalized_mutual_info`:** emparejaba cada celda con la marginal de otra, porque
  los joins reordenaban las filas, y podía devolver valores mayores que 1. El benchmark lo
  destapó al marcar 1,308 contra el grupo de MCC; ahora marca exactamente 1.
- **Tests nuevos:** el protocolo de fuga (barajar el objetivo de la mitad de prueba no mueve los
  segmentos), la independencia de la evaluación respecto al orden de filas, la información
  marginal sobre señal y ruido, y los límites de la NMI.

### Resultado (datos sintéticos, semilla 20260910)

| Método | Segmentos | R² fuera de muestra | Lift decil | NMI con MCC | NMI con clientela |
|---|---|---|---|---|---|
| Grupo de MCC, sin modelo | 12 | 0,740 | 3,63 | 1,000 | 0,033 |
| k-means, atributos | 10 | 0,081 | 1,55 | 0,213 | 0,059 |
| k-means, grafo | 12 | 0,011 | 1,07 | 0,035 | **0,373** |
| k-means, atributos + grafo | 12 | 0,069 | 1,52 | 0,200 | 0,220 |
| Leiden | 8 (55 % cobertura) | 0,001 | 0,83 | 0,008 | 0,002 |
| **Supervisada, atributos** | 12 | **0,751** | **3,81** | 0,838 | 0,033 |
| Supervisada, atributos + grafo | 12 | 0,751 | 3,81 | 0,838 | 0,033 |

Información marginal con el mismo predictor: atributos 0,751; solo grafo 0,105; atributos +
grafo 0,751, una diferencia de 0,000.

- **La segmentación supervisada es la que sirve** y viene con doce reglas legibles.
- **El grafo no aporta nada sobre los atributos** para este objetivo, ni siquiera con un
  predictor fuerte. La conclusión de la Feature 4 no cambia: se refuerza.
- **El grafo sí encuentra la estructura que buscaba** (NMI 0,373 con la clientela plantada).
  Esa estructura decide a dónde se va el volumen cuando un comercio deja de aceptar, que es
  justo lo que mide la Feature 5.

### Decisiones

- **Escalera completa** (decisión tuya): benchmark, supervisada e información marginal, en vez
  de quedarse solo con la medición o solo con el arreglo de escalado.
- **PR aparte** (decisión tuya): la Feature 4 se mergeó primero y esta revisión metodológica va
  en su propio PR.
- **El número de hojas se elige dentro de la mitad de entrenamiento**, nunca sobre la de prueba.
- **Los segmentos supervisados sirven al objetivo con el que se cortaron:** para preguntas de
  sustitución hay que volver a la representación del grafo.

### Pendientes

- **Leiden sigue siendo el más débil** y no se ajustó para favorecerlo: con menos poda subiría
  la cobertura.
- **El grafo merece una segunda audiencia en la Feature 5**, con la curva de abandono y el link
  prediction.
- Siguen de features anteriores: contrastar la tabla de interchange con Mastercard y Redeban;
  anclas de BanRep 2025; el `.venv` local en 3.11.9; `CLAUDE.md` fuera del repo; cerrar los
  notebooks que tengan abierto el warehouse antes de `dbt build`.

## Feature 4 — Segmentación con Graph ML (mergeada, PR #5)

`src/ips/graph/`: grafo bipartito tarjetahabiente-comercio con ventana y proyección podada,
embeddings espectrales (PPMI + SVD truncado), perfil del comercio, las segmentaciones no
supervisadas (atributos, grafo, ambos y comunidades de Leiden), la evaluación con su frase de
veredicto, los perfiles de segmento y `python -m ips.tasks segment`. Extra `graph` con
scikit-learn y scipy, notebook 03 y supuestos F4-01 a F4-09 en `docs/assumptions.md`:

- el grafo no le gana al baseline de atributos en la métrica de negocio, y así se reportó;
- el grafo sí recupera la clientela que plantó el generador (NMI 0,373 contra 0,059);
- dos corridas separadas dan segmentos idénticos para los 28.438 comercios.

## Feature 3 — Unit economics y revenue analytics (mergeada, PR #4)

P&L en dos capas: la cascada de tarifas que cumple la conservación (emisor bruto + red +
adquirente bruto = MDR) y los costos operativos que dan la contribución de cada actor, con
precio blended o IC++ por comercio. Métricas (yield, interchange efectivo, contribución del
emisor, carga relativa), revenue bridge por Shapley y el notebook 02. Supuestos F3-01 a F3-13
en `docs/assumptions.md`:

- yield de la red de 28,1 pb; de cada 100 COP de MDR, 69,2 al emisor, 13,4 a la red y 17,4 al
  adquirente;
- margen bruto del adquirente de 0,45 % en blended y 0,26 % en IC++;
- recargo de 1,5 pp por tarjeta extranjera y tarifas blended recalibradas por el fee fijo de
  autorización.

## Feature 2 — Pipeline y datasets analíticos (mergeada, PR #3)

Proyecto dbt sobre DuckDB (fuentes, staging, `fct_transactions`, dimensiones y 4 marts) con el
test de conservación en SQL, la capa de I/O (`utils/io.py`), el runner `python -m ips.tasks` y
CI con Python 3.11 y 3.12. Supuestos F2-01 a F2-06 en `docs/assumptions.md`:

- dbt corre en un subproceso, con las rutas en `IPS_RAW_DIR` e `IPS_WAREHOUSE`;
- dbt-duckdb va en el extra `pipeline`;
- `fct_transactions` coincide con Python fila a fila en la muestra.

## Feature 1 — Generador de datos sintéticos (mergeada, PR #2)

6M compras en 24 meses para una red colombiana ficticia, calibradas con Superfinanciera,
BanRep, DANE y BCE, con los supuestos F1-01 a F1-30 en `docs/assumptions.md`:

- la F0 quedó retirada;
- se usa igraph en lugar de networkx;
- las etiquetas latentes se guardan aparte en `data/raw/latent/`;
- el notebook `01_data_validation.ipynb` trae la tabla de calibración.

## Feature 0 — Vertical slice (mergeada, PR #1; retirada en la Feature 1)

Tajada vertical con un generador simple, P&L por actor y test de conservación. Su P&L
transitorio vivió hasta la Feature 3, que lo reemplazó por `ips.economics.pnl`.
