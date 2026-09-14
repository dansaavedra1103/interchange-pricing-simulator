# Estado del proyecto

## Feature en curso

La Feature 6 (simulador de escenarios) está cerrada en la rama `feat/06-simulador`, con PR
contra `main`. Siguiente según el orden de CLAUDE.md: **Feature 9 — App, en borrador**, que
mostrará los escenarios; después vienen la 8 (forecasting), la 7 (optimización) y la 9 final.

## Feature 6 — Simulador de escenarios (cerrada)

### Por qué

Las features 1 a 5 dejaron cada pieza por separado: un P&L que se conserva, una curva de abandono
calibrada y los sustitutos de GraphSAGE. Faltaba el motor que las encadena para responder lo que
un equipo de pricing pregunta antes de tocar una tarifa: **si esto cambia, ¿quién termina
pagándolo?** Es la función que la Feature 7 va a optimizar y lo que la Feature 9 va a mostrar.

### Qué se hizo

- **Escenarios en YAML** (`scenario.py`, `config/scenarios/`): cuatro palancas —tope de
  interchange, descuento, sustitución de volumen y migración de producto—, validadas con pydantic
  antes de tarificar nada. Seis escenarios estándar y el nulo como control.
- **Estado base** (`state.py`): 2025 tarificado una vez, la curva de abandono calibrada sobre esa
  misma población, el abandono de la base y los sustitutos que aprendió la Feature 5.
- **Motor** (`engine.py`): una pasada por la cadena de la spec —tarifa, traslado a la tarifa
  blended, abandono adicional, redistribución con signo, recompensas, gasto y P&L—, con cada
  efecto de volumen como un peso por transacción.
- **Resultado** (`result.py`): totales, P&L por actor y por segmento, revenue bridge contra la
  base y una frase por escenario. La corrida se detiene si las tarifas dejan de sumar el MDR o si
  el volumen perdido deja de cuadrar con el movido más la fuga.
- **Sensibilidad** (`sensitivity.py`): barrido de un parámetro y tornado sobre cualquier ruta de
  la configuración, con la curva recalibrada en cada valor.
- **Un gancho mínimo en `compute_pnl`:** `interchange_adjustment`, aplicado justo después de la
  tabla. Sin ajuste, el resultado es idéntico al de antes.
- **Sumas agrupadas estables** (`utils/frames.py`): el test de determinismo encontró que dos
  corridas del mismo escenario diferían en los últimos bits. La causa: polars 1.44 puede sumar
  los flotantes de un grupo en otro orden en cada llamada (13 de 300 repeticiones sobre grupos de
  pocas filas). Toda suma agrupada que alimenta un resultado pasa ahora por `stable_group_sums`,
  que suma fila por fila en el orden del frame. Eso tocó también el revenue bridge (Feature 3) y
  la redistribución (Feature 5), solo en el orden de las sumas.
- `python -m ips.simulator run --scenario <archivo>` o `--all`, `python -m ips.tasks simulate`,
  `make simulate`, un paso de CI, `config/simulator.yaml`, el notebook 05 y los supuestos F6-01 a
  F6-15.
- **Tests** (`tests/test_simulator.py`, `tests/test_frames.py`, un caso nuevo en
  `test_revenue_bridge.py` y otro en `test_config.py`): escenario nulo = base, conservación de
  tarifas y de volumen en cada escenario, seis corridas idénticas, cada palanca exacta, traslados
  en 0 y en 1, recompensas con piso en cero y sin aparecer donde no las hay, validación de
  escenarios y el gancho del P&L.

### Resultado (datos sintéticos, semilla 20260910)

Los seis escenarios y el nulo corren en 36 segundos sobre las 3,26M compras de 2025 (el año base
se tarifica y calibra en 2), y dos corridas separadas dan sus 43 artifacts idénticos bit a bit.
Cambio frente al año base:

| Escenario | Ingreso de la red | Contribución del emisor | Contribución del adquirente | Costo del comercio | Volumen en tarjeta |
|---|---|---|---|---|---|
| Tope al débito en 0,3 % | +0,39 % | −41,18 % | +46,95 % | −8,24 % | +0,33 % |
| Tope al crédito de consumo en 0,5 % | +0,79 % | −50,45 % | +105,15 % | −17,16 % | +0,38 % |
| Pago instantáneo: 20 % del débito bajo 50.000 COP | −2,49 % | −0,48 % | −0,19 % | −0,81 % | −1,05 % |
| El comercio más grande, con 25 % menos de interchange | −0,00 % | −1,19 % | +0,00 % | −0,63 % | +0,01 % |
| 10 % del crédito estándar pasa a premium | −0,03 % | −0,50 % | −1,42 % | +0,24 % | −0,02 % |
| CNP tokenizado sin el recargo de no presente | +0,07 % | −5,70 % | +9,08 % | −1,56 % | +0,05 % |

- **Un tope lo pagan los emisores, y con traslados del 50 % lo reparten comercios y
  adquirentes.** Con el tope al crédito, la contribución del emisor cae a la mitad, los comercios
  pagan 17 % menos y la contribución de los adquirentes se duplica: la tarifa blended solo baja la
  mitad de lo que baja el interchange.
- **El pago instantáneo es el único escenario que le quita ingreso a la red, y le pega más que al
  volumen** (−2,49 % contra −1,05 %). La red cobra 120 COP fijos por compra además del 0,15 % del
  monto, y en una compra pequeña ese fijo pesa más.
- **Un descuento a un comercio en IC++ pasa entero del emisor al comercio.** El comercio más
  grande del año base cotiza en IC++: su descuento no mueve a la red ni al adquirente, y lo
  financia el emisor.
- **Pasar crédito estándar a premium le cuesta al emisor:** el interchange sube entre 0,3 y
  0,5 pp más 50 COP por compra, pero las recompensas suben 0,8 pp. Los adquirentes absorben la
  mitad del alza que no trasladan a sus comercios blended.
- **La red gana con los topes por el volumen y por dónde cae.** Sus fees no dependen del
  interchange, pero los comercios que retienen proporcionalmente más volumen son los que tienen
  más compras cross-border: con el tope al crédito, ese volumen crece 2,7 % frente a 0,6 % en las
  domésticas, y la red cobra 143 pb del monto en una compra cross-border contra 22 pb en una
  doméstica. De los 12,1 millones de COP que gana, 5,9 vienen del volumen y 4,6 del peso
  cross-border.
- **Si la red gana con un tope depende de los adquirentes.** Con el tope al crédito, la red gana
  0,79 % cuando los adquirentes trasladan la mitad a su tarifa blended, 1,40 % si trasladan todo y
  pierde 0,04 % si no trasladan nada: el volumen que gana viene de comercios que ven bajar su
  tarifa. Es el supuesto que más mueve el resultado. Le siguen la fuga (+0,26 % con 20 %, +1,31 %
  con 50 %: más fuga significa que el volumen que los comercios retienen viene del efectivo y no
  de otros comercios) y la segunda ancla de la curva (+0,17 % a +1,00 %). El exponente de los
  niveles por grupo casi no pesa (+0,79 % a +0,89 %).
- **Trasladar un tope a las recompensas amortigua al emisor, hasta que se acaban:** su
  contribución cae 88 % sin traslado y 45 % con un traslado de 75 % o más, cuando las
  recompensas del crédito ya están en cero. A cambio, los titulares gastan menos y la red gana
  0,72 % en vez de 1,20 %.
- **Una consecuencia de la definición de carga, no un hallazgo:** en el pago instantáneo el
  abandono sube (214 millones de COP de volumen) aunque la factura de los comercios baje. Sale
  débito, la tarjeta más barata, y sube la tasa promedio de lo que queda en tarjeta, que es lo que
  mide la carga relativa.

### Decisiones

- **Año observado + cambio** (decisión tuya): un escenario reescribe 2025 con otros precios y le
  cobra a cada comercio solo el abandono adicional que causan. El escenario nulo reproduce la base
  exacta.
- **Traslados del 50 %, barridos de 0 a 100 %** (decisión tuya): cuánto del cambio de interchange
  pasa a la tarifa blended y cuánto a las recompensas. Los comercios en IC++ siempre ven el cambio
  completo.
- **Abandono en valor esperado** (decisión tuya): determinista y suave, como lo necesita la
  Feature 7. Queda como desviación documentada de la spec, que describe un simulador estocástico.
- **El tope, separado en débito y crédito**, como sugiere el ejemplo de CLI de la spec; el crédito
  comercial queda exento, como en la IFR.
- **Pesos por transacción, no filas nuevas ni borradas:** la identidad de tarifas se cumple fila a
  fila con cualquier volumen.
- **Una sola pasada, sin equilibrio**, y **recompensas por producto**, la granularidad de
  `economics.yaml`.
- **Desvíos del plan:** `cardholder_response.py` no cambió (el motor la llama una vez por
  producto); el test del gancho de `compute_pnl` quedó en `test_simulator.py`; tres escenarios
  tomaron los nombres del árbol de la spec (`instant_payments_entry`, `strategic_merchant_deal` y
  `cnp_tokenized_discount`), que el plan había cambiado sin advertirlo; y aparecieron las sumas
  agrupadas estables, que el plan no preveía.

### Pendientes

- **Efectos de segundo orden:** los sustitutos que reciben volumen no vuelven a pasar por la
  curva, y nadie reprecia en respuesta al escenario.
- **Una versión estocástica** si la app necesita colas o intervalos además del valor esperado.
- **Fuente para los traslados**, hoy [P]; el notebook 05 muestra cuánto mueven las conclusiones.
- **La carga como tasa promedio:** un escenario que saca la tarjeta más barata sube el abandono
  aunque la factura baje. Vale la pena contrastarla con una carga sobre la factura total.
- **`merchant_profile` sigue sumando flotantes con `group_by`:** esas sumas no llegan al
  simulador, pero podrían variar en los últimos bits en la segmentación.
- Siguen de la Feature 5: fuente para las anclas de la curva y las semi-elasticidades de
  recompensas; presupuesto de épocas del GNN; la probabilidad de abandono como columna de
  `graph/evaluate.py`; sustitutos comercio a comercio.
- Siguen de antes: Leiden sigue siendo el más débil; contrastar la tabla de interchange con
  Mastercard y Redeban; anclas de BanRep 2025; el `.venv` local en 3.11.9; `CLAUDE.md` fuera del
  repo; cerrar los notebooks que tengan abierto el warehouse antes de `dbt build`.

## Feature 5 — Elasticidad y link prediction (mergeada, PR #7)

### Por qué

El simulador necesita dos respuestas antes de tocar una tarifa: **cuánto** volumen se va cuando
sube la carga de un comercio, y **a dónde** se va. Sin la primera, la Feature 7 no tiene función
objetivo; sin la segunda, el volumen perdido desaparece en vez de moverse a otros comercios.

### Qué se hizo

- **Curva de aceptación** (`merchant_acceptance.py`): logística sobre la carga relativa, con
  modificadores por canal, recargo permitido, presión del pago instantáneo y modelo de precio.
- **Calibración por momentos** (`calibration.py`): bisección anidada que resuelve un intercepto
  por grupo de MCC y una pendiente común para reproducir dos anclas sobre la distribución real de
  la población. El nivel de cada grupo es proporcional a la mediana de su carga relativa. Se
  calibra sobre la curva ya acotada, y el intervalo de cada intercepto se amplía solo, para no
  devolver un extremo en silencio.
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
- **Tests** (`tests/test_elasticity.py` y cuatro casos nuevos en `test_config.py`): monotonicidad
  y modificadores de la curva, calibración, niveles por grupo, costo de aceptación en todos los
  grupos, respuesta del titular, protocolo sin fuga, determinismo del GNN, intervalos, sustitutos
  por bloques y conservación del volumen.

### Resultado (datos sintéticos, semilla 20260910)

La corrida completa toma ~8 minutos (7,5 de ellos en GraphSAGE) y dos corridas separadas dan
artifacts idénticos bit a bit. La curva reproduce las dos anclas exactamente, con una pendiente
común β = 12,53 e interceptos que llevan a cada grupo a su nivel, de 1,1 % en educación a 6,3 %
en mercado. Entrenan 40.000 titulares y se evalúan 2.000, sobre 5.047 enlaces nuevos.

| Calibración | β | Comercios en el piso | Grupos sin respuesta* | Volumen en riesgo en los dos grupos que más concentran |
|---|---|---|---|---|
| Un solo nivel (primera versión) | 34,01 | 87 % | 8 de 12 | 88 % (mercado y combustible) |
| **Nivel por grupo, proporcional a la carga** | **12,53** | **5 %** | **1 de 12** | 52 % (viajes y electrónica) |

\* Menos de 0,1 pp de respuesta a un MDR 10 % más alto.

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
- **La curva ya no es un umbral.** Con un nivel por grupo, solo el 5 % de los comercios queda en
  el piso de 0,05 % (antes, el 87 %) y todos los grupos responden a un MDR más alto: de +0,06 pp
  en educación, cuyo MDR casi no le pesa, a +4,8 pp en mercado, ponderado por volumen.
- **El volumen en riesgo se reparte distinto:** viajes y electrónica suman el 52 % —mucho volumen
  con carga media o alta—, y mercado y combustible, que antes concentraban el 88 %, bajan al 19 %.
- **Los sustitutos de GraphSAGE comparten el grupo de MCC el 99 % de las veces**, y la
  redistribución cuadra al peso: de 39.779 millones de COP perdidos, 25.843 millones pasan a
  sustitutos y 13.936 millones salen del riel de tarjetas.

### Decisiones

- **Calibración por momentos** (decisión tuya), no parámetros a mano.
- **Escalera evaluada** (decisión tuya), con partición temporal y baselines sin grafo.
- **Traer torch** (decisión tuya): GraphSAGE en PyG, sin `NeighborLoader`, así que no hacen
  falta `pyg-lib` ni `torch-sparse`, que en Windows exigen compilación.
- **Nivel de abandono por grupo, proporcional a su carga** (decisión tuya). La primera versión
  usaba un solo intercepto para toda la población, y la pendiente que exigía la segunda ancla
  volvía la curva un umbral: siete de los doce grupos quedaban en el piso, sin respuesta al
  precio. Se midieron tres variantes sobre los 6M —mismo nivel para todos, proporcional a la
  carga y proporcional a su raíz— y las tres lo corregían; la proporcional conserva, también entre
  sectores, la premisa de que más carga significa más abandono.
- **Las anclas quedaron marcadas [P], no [L]** como decía el plan: no hay una fuente publicada
  verificada con esas cifras, y no se inventan citas.
- **Las épocas las elige la validación**, nunca los meses reservados. Un primer barrido que las
  elegía mirando la prueba se descartó.
- **Una ventaja solo cuenta si su intervalo no cruza el cero.**

### Pendientes

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
