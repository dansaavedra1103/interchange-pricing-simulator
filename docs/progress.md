# Estado del proyecto

## Feature en curso

La Feature 3 (unit economics y revenue analytics) está cerrada en la rama
`feat/03-unit-economics`, con PR contra `main`. Siguiente según el orden de CLAUDE.md:
**Feature 4 — Graph ML**, que debe ganarle a un k-means sobre atributos o reportar que no.

## Feature 3 — Unit economics y revenue analytics (cerrada)

### Qué se hizo

- **`config/economics.yaml`**, incluido desde `base.yaml`, con sus modelos en
  `utils/config.py` (`EconomicsConfig`):
  - los fees de la red a cada lado: assessment, autorización y cross-border;
  - el margen IC++, el recargo blended por tarjeta extranjera y el procesamiento del
    adquirente;
  - los costos del emisor por producto.
- **`src/ips/economics/`:**
  - `pnl.py` reemplaza a `simple_pnl.py`. `NetworkPnL`, `MerchantCost`, `IssuerPnL` y
    `AcquirerPnL` arman la cascada de tarifas y los costos operativos. `compute_pnl` devuelve
    el P&L por transacción y por actor, y `pnl_by` lo agrega por cualquier segmento;
  - `metrics.py`: net revenue yield con sus drivers, interchange efectivo, contribución del
    emisor y carga relativa;
  - `revenue_bridge.py`: reparte el cambio del ingreso de la red entre dos períodos o
    escenarios en volumen, participación cross-border, mezcla y tarifa, con valores de
    Shapley;
  - `params.py`: las tablas de referencia que lee el warehouse.
- **Generador:** `merchants.pricing_model` (blended o IC++) según el adquirente y el tramo del
  comercio. No consume números aleatorios, así que el resto de los datos no cambia.
- **dbt:**
  - fuente y staging de `pricing_terms`;
  - `fct_transactions` con la cascada completa y los drivers del ingreso de la red;
  - marts con emisor bruto, red y adquirente bruto;
  - el test de conservación verifica también los drivers de la red;
  - `python -m ips.tasks dbt` reescribe las tablas de referencia antes de construir.
- **Calibración:** tarifas blended recalibradas por el fee fijo de autorización; margen IC++
  de 0,20 % + 100 COP.
- **Tests:** 88 rápidos (64 antes) y 2 lentos.
  - `test_economics.py`: conservación contra un MDR recalculado por fuera y en 11
    agregaciones, ejemplos calculados a mano (spec §1.2, IC++, micropago, cross-border,
    premium con fraude con y sin contracargo), métricas, errores y rangos de calibración;
  - `test_revenue_bridge.py`: los efectos suman el cambio, entradas iguales dan cero y cada
    efecto puro aparece solo;
  - `test_pipeline.py`: paridad SQL ↔ Python fila a fila en todas las columnas de la cascada.
- **Notebooks:** `02_unit_economics.ipynb` (nuevo) con 7 gráficos y sus tablas. El 01 migra
  al nuevo P&L y suma el yield de la red a la tabla de calibración.
- Supuestos F3-01 a F3-13 en `docs/assumptions.md`; `docs/architecture.md` actualizado.

### Resultado (datos sintéticos, semilla 20260910)

- **Yield de la red: 28,1 pb del volumen** (15,0 de assessment, 7,2 de autorización y 5,9
  cross-border). En 2025 Visa y Mastercard ingresaron ~29–31 pb netos de incentivos y ~37–38
  pb brutos; el modelo no tiene incentivos ni servicios de valor agregado.
- **De cada 100 COP de MDR:** 69,2 van al emisor, 13,4 a la red y 17,4 al adquirente. Tras
  costos, el emisor conserva 33,4 y el adquirente 9,8.
- **Adquirentes:** margen bruto de 0,45 % en blended doméstico (0,43–0,47 % por grupo) y de
  0,26 % en IC++. Sobre las mismas compras, IC++ sale ~0,17 pp más barato en cada tramo.
- **Emisores:** contribución de 0,69 % del volumen en débito y crédito estándar, 0,52 % en
  premium y 1,08 % en comercial.
- **Carga relativa:** la mediana llega a ~50 % del margen en alimentos, gasolina y viajes.
- **Revenue bridge 2024 → 2025:** +244M COP, 99 % por volumen.
- **`dbt build` sobre los 6M:** 80 de 80 nodos OK en ~16 s. Python y SQL coinciden fila a fila
  en la muestra.

### Decisiones

- **IC++ desde ya** (decisión tuya): los bancos lo cobran a sus comercios grandes y medianos.
- **Parámetros en `config/economics.yaml`** (decisión tuya).
- **Montos en decimales** (decisión tuya): la conservación se cumple con tolerancia 1e-9. Cierra
  el pendiente de liquidar en centavos.
- **La contribución del emisor es economía de pagos**, sin intereses ni cartera revolvente.
- **El modelo de precio es un dato del comercio** (`pricing_model`): Python y SQL leen la misma
  columna, así que la regla no se duplica.
- **SQL cubre la cascada** (la identidad de conservación) y Python suma los costos.
- **Recargo blended de 1,5 pp por tarjeta extranjera**, nuevo respecto al plan: sin él, el
  adquirente extranjero perdía ~1,3 % en cada compra premium cross-border. Los comercios del
  exterior siguen en blended, como en el plan.
- **Margen IC++ más bajo que el del plan** (0,20 % + 100 COP en vez de 0,35 % + 200): con el del
  plan, IC++ les costaba a los comercios grandes lo mismo que blended.
- **Assessment del emisor de 0,02 %** (el plan decía 0,05 %): con el 0,13 % del adquirente, los
  assessments de ambos lados suman 0,15 %, el techo que fija la spec (§1.6).
- **Gráficos del notebook 02:** la forma la decidió la skill de dataviz. El reparto del MDR es
  una barra apilada (parte de un todo) y no una cascada; la contribución del emisor es un
  dumbbell (bruto → contribución) y no barras divergentes. Se sumó un gráfico del margen del
  emisor por tamaño de ticket.

### Pendientes

- **Micropagos:** en el tramo reducido, el fee fijo de autorización del emisor supera al
  interchange, y el emisor pierde antes de costos en ~22 % de las compras con débito. Es una
  palanca para el optimizador.
- **Transporte:** con tickets de ~16.000 COP, el procesamiento fijo deja al adquirente con
  contribución negativa.
- **Adquirente extranjero:** margen bruto de ~0,1 % en la parte de su negocio con tarjetas
  colombianas, la única que ve el modelo.
- No se modelan incentivos a clientes, servicios de valor agregado ni costos de la red.
- Siguen de la F2:
  - contrastar la tabla de interchange con las de Mastercard y Redeban (Decreto 1692 de 2020);
  - actualizar las anclas de calibración con las cifras de 2025 de BanRep;
  - el `.venv` local sigue en Python 3.11.9 (la CI cubre 3.12);
  - `CLAUDE.md` vive en `C:\Users\USUARIO\` y no en el repo;
  - DuckDB admite un escritor o varios lectores: hay que cerrar los notebooks que tengan
    abierto el warehouse antes de correr `dbt build`.

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
