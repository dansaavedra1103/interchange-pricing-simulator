# Estado del proyecto

## Feature en curso

La Feature 2 (pipeline y datasets analíticos) está cerrada en la rama `feat/02-dbt-pipeline`,
con PR contra `main`. Siguiente según el orden de CLAUDE.md: **Feature 3 — Unit economics y
revenue analytics**.

## Feature 2 — Pipeline y datasets analíticos (cerrada)

### Qué se hizo

- **Proyecto dbt en `dbt/`, sobre DuckDB:**
  - fuentes `raw` que leen los parquet en su lugar;
  - 8 vistas de staging;
  - `fct_transactions`, con interchange, scheme fee, MDR y margen del adquirente por fila;
  - 5 dimensiones;
  - 4 marts: P&L por actor, interchange efectivo, carga relativa por comercio y aristas
    mensuales del grafo;
  - la macro `interchange_lookup`;
  - el test singular `assert_pnl_conservation`.
- **`src/ips/utils/io.py`:**
  - rutas desde el config y lectura de parquet;
  - el warehouse se abre en solo lectura por defecto;
  - `run_dbt` corre dbt en un subproceso, con las rutas pasadas por variable de entorno.
- **`src/ips/tasks.py`:** `python -m ips.tasks generate|dbt|docs|test|all`, multiplataforma.
  El `Makefile` delega en él.
- **CI** (`.github/workflows/ci.yml`) con Python 3.11 y 3.12:
  1. ruff y pytest;
  2. generación de la muestra;
  3. `dbt build`.
- **Configuración:**
  - `sample_sizes` y `output.warehouse_path` en `base.yaml`;
  - `generate --sample`;
  - `network_fees.parquet`.
- **Tests:** 64 rápidos.
  - `test_pipeline.py` construye el warehouse sobre la muestra y lo compara con Python fila a
    fila y por mart.
  - `test_io.py` cubre la capa de I/O.
- `docs/architecture.md`, con el diagrama de linaje; supuestos F2-01 a F2-06 en
  `docs/assumptions.md`.

### Resultado (datos sintéticos, semilla 20260910)

- **`dbt build` sobre los 6M:** 69 de 69 nodos OK (10 tablas, 8 vistas y 51 tests, incluida
  la conservación) en 13 s. La generación tarda ~7 s.
- **Paridad Python ↔ SQL:** `fct_transactions` coincide con `compute_pnl` en cada
  transacción de la muestra. Los marts cuadran con el fct, y las aristas con
  `diagnostics.edge_list`.
- **`dbt docs generate`:** produce el linaje completo, de las fuentes a los marts.

### Decisiones

- **dbt corre en un subproceso**, no con `dbtRunner` en proceso: así su conexión a DuckDB
  nunca queda abierta en quien lo llama, que puede leer el warehouse enseguida.
- **Rutas absolutas por `IPS_RAW_DIR` e `IPS_WAREHOUSE`:** dbt no depende del directorio de
  trabajo. Para moverse a Snowflake basta cambiar `profiles.yml` y las fuentes.
- **dbt-duckdb va en el extra `pipeline`**, porque el simulador no lo necesita.
- **Runner de tareas en Python** (decisión tuya), porque `make` no existe en Windows.
- **CI con Python 3.11 y 3.12** (decisión tuya): valida la versión que pide la spec.
- **Marts:** las aristas del grafo quedan por mes, y la carga relativa solo para comercios
  con compras.

### Pendientes

- Contrastar la tabla de interchange con las que publican Mastercard y Redeban en Colombia
  (Decreto 1692 de 2020). Hoy devuelven 403.
- Actualizar las anclas de calibración con las cifras de 2025 del reporte de BanRep.
- El `.venv` local sigue en Python 3.11.9; la CI cubre 3.12.
- `CLAUDE.md` vive en `C:\Users\USUARIO\` y no en el repo.
- Evaluar en la F3 liquidar en unidades enteras (centavos), para que la conservación y el
  determinismo sean exactos sin tolerancia.
- **DuckDB** permite un escritor o varios lectores, no ambos a la vez: hay que cerrar los
  notebooks que tengan abierto el warehouse antes de correr `dbt build`.

## Feature 1 — Generador de datos sintéticos (mergeada, PR #2)

6M compras en 24 meses para una red colombiana ficticia, calibradas con Superfinanciera,
BanRep, DANE y BCE, con los supuestos F1-01 a F1-30 en `docs/assumptions.md`:

- la F0 quedó retirada;
- se usa igraph en lugar de networkx;
- las etiquetas latentes se guardan aparte en `data/raw/latent/`;
- el notebook `01_data_validation.ipynb` trae la tabla de calibración.

## Feature 0 — Vertical slice (mergeada, PR #1; retirada en la Feature 1)

Tajada vertical con un generador simple, P&L por actor y test de conservación. El P&L
transitorio y su test siguen vivos sobre los datos de la Feature 1.
