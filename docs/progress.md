# Estado del proyecto

## Feature en curso

La Feature 4 (segmentación con Graph ML) está cerrada en la rama `feat/04-graph-segmentation`,
con PR contra `main`. Siguiente según el orden de CLAUDE.md: **Feature 5 — Elasticidad**, que
traerá la curva de abandono de aceptación y el link prediction.

## Feature 4 — Segmentación con Graph ML (cerrada)

### Qué se hizo

- **`src/ips/graph/`**, que lee el warehouse y escribe artefactos, nunca al revés:
  - `build_graph.py`: ventana de compras, matriz dispersa tarjetahabiente × comercio y
    proyección comercio-comercio podada a los comercios habituales de cada tarjeta;
  - `features.py`: perfil del comercio (atributos, mezcla de clientes y los dos objetivos:
    interchange efectivo y carga relativa), con una sola definición para producción y tests;
  - `spectral_embed.py`: PPMI sobre el grafo bipartito y SVD truncado;
  - `baseline_kmeans.py`, `clustering.py`, `communities.py`: las cuatro segmentaciones
    (atributos, embeddings, ambos, comunidades de Leiden), con k elegido por silueta;
  - `evaluate.py` y `segment_profiles.py`: la tabla comparativa, la frase de veredicto y el
    perfil de negocio de cada segmento;
  - `pipeline.py`: orquestación, artefactos en `data/artifacts/`.
- **`python -m ips.tasks segment`** (y `make segment`) corre todo y escribe embeddings,
  segmentos, comparación y perfiles.
- **Configuración:** sección `segmentation` en `base.yaml` con ventana, mínimo de compras,
  dimensiones, rango de k, poda de la proyección y fracción de prueba.
- **Dependencias:** extra `graph` con scikit-learn y scipy; la CI instala `.[dev,pipeline,graph]`.
- **Tests:** 18 nuevos en `tests/test_graph.py` (108 rápidos en total), todos sobre la muestra y
  sin dbt: grafo bien formado, ventana, poda, embeddings deterministas y que separan clientelas
  plantadas, clustering reproducible, η² y R² fuera de muestra sobre casos de juguete, Leiden
  contra etiquetas barajadas, perfiles y artefactos.
- **`notebooks/03_segmentation_comparison.ipynb`** con la comparación y la conclusión explícita.
- Supuestos F4-01 a F4-09 en `docs/assumptions.md`; `docs/architecture.md` actualizado.

### Resultado (datos sintéticos, semilla 20260910)

28.438 comercios segmentados de 30.521 con compras; 5,8M aristas en 24 meses; la corrida
completa toma ~35 s.

| Método | Segmentos | η² interchange | η² carga | R² fuera de muestra | NMI clientela |
|---|---|---|---|---|---|
| Atributos (baseline) | 8 | 0,497 | 0,078 | **0,077** | 0,084 |
| Grafo (embeddings) | 12 | 0,403 | 0,015 | 0,011 | **0,373** |
| Híbrido | 12 | **0,583** | 0,073 | 0,069 | 0,212 |
| Leiden | 8 (55 % de cobertura) | 0,045 | 0,002 | 0,001 | 0,002 |

- **El grafo no le gana al baseline en la métrica de negocio** y así se reporta: para predecir
  la carga relativa de un comercio que el modelo no vio, los atributos explican 0,077 y el
  grafo 0,011. La razón es económica: la carga es MDR efectivo sobre margen sectorial, y el
  margen es una propiedad del grupo de MCC, justo lo que el baseline codifica.
- **El grafo sí aporta donde está el ingreso:** sumado a los atributos, sube la homogeneidad
  del interchange efectivo de 0,497 a 0,583, casi 9 puntos, porque agrupa comercios que
  comparten clientes y por eso comparten mezcla de tarjetas.
- **El método funciona:** solo con el grafo, la segmentación recupera la clientela plantada por
  el generador (NMI 0,373 contra 0,084 del baseline). Es diagnóstico del generador, no
  evidencia sobre el mercado real.
- **Leiden sobre la proyección podada es el más débil:** deja fuera al 45 % de los comercios y
  explica casi nada; la poda a comercios habituales conserva señal de clientela pero pierde
  cobertura.
- **El veredicto aguanta el barrido de k:** entre 4 y 12 segmentos el grafo siempre queda
  debajo del baseline. De paso aparece que la silueta no elige el k que le sirve al negocio: el
  baseline pasa de 0,077 con k = 8 a 0,166 con k = 10.
- **Las corridas son reproducibles:** dos ejecuciones separadas dan segmentos idénticos para
  los 28.438 comercios.

### Decisiones

- **Embeddings espectrales, no node2vec** (decisión tuya): PPMI + SVD truncado es la
  factorización matricial que las caminatas aleatorias aproximan (Qiu et al., 2018). Sin torch
  ni gensim, determinista y en segundos, así que la CI lo corre.
- **La carga relativa como sustituto del abandono** (decisión tuya): es el predictor que la
  spec nombra. Cuando la Feature 5 traiga la curva, `evaluate.py` suma la métrica real como una
  columna más.
- **Segmentos como artefactos** (decisión tuya) en `data/artifacts/segments/`, no en el
  warehouse: `dbt build` no depende de que un modelo haya corrido.
- **Sin mart nuevo:** el perfil del comercio se calcula una sola vez en Python, así los tests
  del grafo corren sobre la muestra sin dbt.
- **Cuarto método, el híbrido** (no estaba en el plan): responde la pregunta que sigue a la
  comparación, si el grafo agrega algo sobre lo que el banco ya sabe. Sin él, la conclusión se
  queda en "no gana" sin decir dónde sí sirve.
- **La proyección se poda a los comercios habituales de cada tarjeta:** sin ese tope genera
  ~10^8 pares y una tarjeta que compra en todas partes conecta todo con todo.
- **La clientela plantada es diagnóstico, nunca criterio:** el veredicto lo decide la métrica
  de negocio fuera de muestra.

### Pendientes

- **Leiden merece otra proyección:** con más vecinos o menos poda subiría la cobertura; hoy es
  el método más débil y no se ajustó para favorecerlo.
- **El híbrido no mejora la carga relativa**, solo el interchange: queda ver en la Feature 5 si
  ayuda a predecir el abandono real.
- **Link prediction** (spec §2.5) es de la Feature 5: estimar el volumen que se redistribuye
  cuando un comercio deja de aceptar.
- Siguen de features anteriores: contrastar la tabla de interchange con Mastercard y Redeban;
  anclas de BanRep 2025; el `.venv` local en 3.11.9; `CLAUDE.md` fuera del repo; cerrar los
  notebooks que tengan abierto el warehouse antes de `dbt build`.

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
