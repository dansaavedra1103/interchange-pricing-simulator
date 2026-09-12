# Estado del proyecto

## Feature en curso

La Feature 5 (elasticidad y link prediction) está cerrada en la rama `feat/05-elasticidad`,
con PR contra `main`. Siguiente según el orden de CLAUDE.md: **Feature 6 — Simulador de
escenarios**, que encadenará la tarifa, el MDR, el abandono, las recompensas, el gasto y el P&L.

## Feature 5 — Elasticidad y link prediction (cerrada)

### Por qué

El simulador necesita dos respuestas antes de tocar una tarifa: **cuánto** volumen se va cuando
sube la carga de un comercio, y **a dónde** se va. Sin la primera, la Feature 7 no tiene función
objetivo; sin la segunda, el volumen perdido desaparece en vez de moverse a otros comercios.

### Qué se hizo

- **Curva de aceptación** (`merchant_acceptance.py`): logística sobre la carga relativa, con
  modificadores por canal, recargo permitido, presión del pago instantáneo y modelo de precio.
- **Calibración por momentos** (`calibration.py`): bisección anidada que resuelve los dos
  parámetros para reproducir dos anclas sobre la distribución real de la población. Se calibra
  sobre la curva ya acotada, y el intervalo del intercepto se amplía solo, para no devolver un
  extremo en silencio.
- **Respuesta del titular** (`cardholder_response.py`): gasto log-lineal en la tasa de
  recompensas, acotado.
- **Escalera de link prediction** (`link_prediction.py`, `scorers.py`, `gnn.py`): partición
  temporal sobre enlaces nuevos y cinco escalones —aleatorio, popularidad global, popularidad
  dentro de las categorías del titular, PPMI+SVD y GraphSAGE en PyTorch Geometric—, con
  recall@10, MRR, cobertura e intervalos bootstrap pareados detrás de cada veredicto.
- **GraphSAGE sin atajos:** aristas divididas en paso de mensajes, ajuste y validación; se
  conserva la época con mejor validación; un hilo y semilla fija.
- **Sustitutos y redistribución:** los 10 vecinos de cada comercio en la representación
  ganadora, calculados por bloques (la matriz completa pesaba ~6 GB), y un reparto del volumen
  perdido que conserva el volumen, fuga incluida.
- `python -m ips.tasks elasticity`, `config/elasticity.yaml`, el extra `gnn` (torch desde el
  índice CPU, también en CI), el notebook 04 y los supuestos F5-01 a F5-17.
- **Tests** (`tests/test_elasticity.py` y tres casos nuevos en `test_config.py`): monotonicidad
  y modificadores de la curva, calibración, respuesta del titular, protocolo sin fuga,
  determinismo del GNN, intervalos, sustitutos por bloques y conservación del volumen.

### Resultado (datos sintéticos, semilla 20260910)

La corrida completa toma ~8 minutos (7,5 de ellos en GraphSAGE) y dos corridas separadas dan
artifacts idénticos bit a bit. La curva reproduce las dos anclas exactamente, con α = −35,06 y
β = 34,01. Entrenan 40.000 titulares y se evalúan 2.000, sobre 5.047 enlaces nuevos.

| Método | recall@10 | Hit rate@10 | MRR | Cobertura |
|---|---|---|---|---|
| Aleatorio | 0,002 | 0,006 | 0,003 | 99,9 % |
| Popularidad global | 0,038 | 0,090 | 0,030 | 0,6 % |
| Popularidad en las categorías del titular | 0,044 | 0,101 | 0,041 | 4,6 % |
| Embedding PPMI+SVD | 0,049 | 0,104 | 0,036 | 74,5 % |
| **GraphSAGE** | **0,053** | **0,118** | **0,048** | 7,8 % |

| Retador frente a referencia | Diferencia en recall@10 | IC 95 % | Veredicto |
|---|---|---|---|
| SVD frente a popularidad por categoría | +0,005 | −0,006 a +0,015 | No se distinguen |
| GraphSAGE frente a popularidad por categoría | +0,008 | +0,002 a +0,015 | **Gana** |
| GraphSAGE frente a SVD | +0,004 | −0,007 a +0,015 | No se distinguen |

- **GraphSAGE es el único escalón que le gana al baseline fuerte sin grafo** con un intervalo que
  no cruza el cero (+0,008 en recall@10, un 19 % relativo), y lidera también en hit rate y MRR.
  El grafo gana su segunda audiencia, por poco.
- **El SVD de la Feature 4 no se distingue de la popularidad por categoría, y GraphSAGE no se
  distingue del SVD.** Sin los intervalos, las dos ventajas se habrían reportado como victorias.
- **Con estas anclas la curva es casi un umbral:** siete de los doce grupos quedan en el piso de
  0,05 % y no responden al MDR, y mercado y combustible concentran el 88 % del volumen en riesgo.
- **Los sustitutos de GraphSAGE comparten el grupo de MCC el 99 % de las veces**, y la
  redistribución cuadra al peso: de 16.322 millones de COP perdidos, 10.602 millones pasan a
  sustitutos y 5.720 millones salen del riel de tarjetas.

### Decisiones

- **Calibración por momentos** (decisión tuya), no parámetros a mano.
- **Escalera evaluada** (decisión tuya), con partición temporal y baselines sin grafo.
- **Traer torch** (decisión tuya): GraphSAGE en PyG, sin `NeighborLoader`, así que no hacen
  falta `pyg-lib` ni `torch-sparse`, que en Windows exigen compilación.
- **Un solo par de parámetros para toda la población**, no uno por grupo como decía el plan:
  calibrar por grupo le daría a todos la misma tasa de abandono. La heterogeneidad entre grupos
  viene de `base_elasticity`.
- **Las anclas quedaron marcadas [P], no [L]** como decía el plan: no hay una fuente publicada
  verificada con esas cifras, y no se inventan citas.
- **Las épocas las elige la validación**, nunca los meses reservados. Un primer barrido que las
  elegía mirando la prueba se descartó.
- **Una ventaja solo cuenta si su intervalo no cruza el cero.**

### Pendientes

- **Revisar el par de anclas antes de la Feature 7:** con 3,5 % y 1,8 pp la curva es casi un
  umbral, y un optimizador vería siete grupos sin costo de aceptación. Las salidas son una
  segunda ancla más suave o calibrar el nivel por grupo; es una decisión de supuestos, no de
  código.
- **Fuente publicada para las dos anclas de la curva** y para las semi-elasticidades de
  recompensas. El framework no cambia: solo `config/elasticity.yaml`.
- **Presupuesto de épocas del GNN en la corrida completa:** con 400 épocas la validación seguía
  mejorando (mejor época, la 392). Un presupuesto mayor solo puede ayudarle a GraphSAGE.
- **Sumar la probabilidad de abandono como columna de `graph/evaluate.py`**, como anticipaba la
  Feature 4.
- **Sustitutos comercio a comercio:** la redistribución exacta repartiría el gasto de cada
  cliente según su propio ranking.
- Siguen de features anteriores: Leiden sigue siendo el más débil; contrastar la tabla de
  interchange con Mastercard y Redeban; anclas de BanRep 2025; el `.venv` local en 3.11.9;
  `CLAUDE.md` fuera del repo; cerrar los notebooks que tengan abierto el warehouse antes de
  `dbt build`.

## Feature 4b — Escalera de comparación y segmentación supervisada (mergeada, PR #6)

Benchmark sin modelo (`segment_by_mcc_group`), segmentación supervisada con una regla legible
por segmento (`supervised.py`), una sola partición para todas las cifras fuera de muestra,
información marginal por representación y el arreglo de `normalized_mutual_info`. Supuestos
F4-10 a F4-14 en `docs/assumptions.md`:

- quedarse con el grupo de MCC, sin modelo, predice la carga relativa fuera de muestra con R²
  0,740, y la segmentación supervisada llega a 0,751;
- con el mismo predictor, el grafo no suma nada sobre los atributos (0,751 contra 0,751);
- el grafo sí recupera la clientela plantada (NMI 0,373), y quedó pendiente darle su segunda
  audiencia en la sustitución, que es la Feature 5.

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
