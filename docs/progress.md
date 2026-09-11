# Estado del proyecto

## Feature en curso

Feature 1 (generador de datos sintéticos) cerrada en la rama `feat/01-synthetic-data`, con PR
contra `main`. Siguiente según el orden de CLAUDE.md: **Feature 2 — Pipeline y datasets
analíticos** (dbt sobre DuckDB).

## Feature 1 — Generador de datos sintéticos (cerrada)

### Qué se hizo

- **Configuración:**
  - `config/base.yaml`, reescrito;
  - `config/mcc_groups.yaml`, con 12 grupos y 33 MCC;
  - `config/interchange_table.yaml`, con la grilla base, los recargos, el tramo de micropagos
    y los overrides.
  - `src/ips/utils/config.py` los carga y valida en conjunto: participaciones que suman 1,
    grilla completa, referencias válidas entre archivos y claves desconocidas rechazadas.
- **`src/ips/data_gen/`:**
  - `entities.py`: contratos pydantic por tabla y los esquemas polars equivalentes;
  - `distributions.py`: los muestreadores;
  - `population.py`: emisores, adquirentes, tarjetas y comercios;
  - `affinity.py`: segmentos latentes, clientela y el muestreador de comercios factorizado;
  - `transactions.py`: el proceso de compras;
  - `fraud.py`: fraude y contracargos;
  - `interchange_table.py`: la tabla expandida, `lookup` y `apply`;
  - `diagnostics.py`: concentración, modularidad, NMI y Leiden;
  - `generate.py`: la CLI, que escribe parquet más `manifest.json` en `data/raw/`.
- `src/ips/utils/logging.py` y `src/ips/economics/simple_pnl.py`, este último adaptado a los
  datos de la Feature 1.
- **Tests:** 55 rápidos (`pytest -q`) y 2 lentos (`pytest -m slow`):
  - `test_config.py`: 12 casos de validación que deben fallar;
  - `test_interchange_table.py`;
  - `test_simple_pnl.py`: la conservación del P&L sobre datos de la Feature 1;
  - `test_data_gen.py`: integridad, determinismo, rangos de calibración y comunidades;
  - lentos: generación a escala completa y ejecución del notebook 01.
- `notebooks/01_data_validation.ipynb`: tabla de calibración, gráficos con matplotlib, grafo
  y chequeo del P&L. Tarda ~60 s sobre los 6M.
- `docs/assumptions.md`: supuestos F1-01 a F1-30 con sus fuentes; la F0 queda marcada como
  retirada.

### Resultado (datos sintéticos, semilla 20260910)

- **Escala:** 6M compras, 150.000 tarjetas y 30.600 comercios, generados en 5,7 s. El parquet
  de transacciones pesa 88,5 MB.
- **Anclas de calibración:**

  | Métrica | Resultado | Ancla |
  |---|---|---|
  | Débito, % de tarjetas | 73,8 % | Superfinanciera: 74 % |
  | Débito, % de compras | 67,6 % | Superfinanciera: 68 % |
  | Ticket promedio | 129k débito, 248k crédito | 131k / 244k |
  | Compras por tarjeta al año | 20 | 19–26 |
  | Crecimiento 2025 vs 2024 | +18,7 % | BanRep: +19 % |
  | Fraude, % del valor | 0,073 % | Spec: 0,05–0,15 % |
  | Fraude en no presente | 83 % del valor | BCE: 84 % |
  | Cross-border vs doméstico | 13,6x | BCE: ~14x |
  | Margen del adquirente | 0,44–0,51 % por grupo | Spec: 0,2–0,7 % |

- **Grafo:** Leiden con resolución 2 recupera la estructura plantada ciudad × clientela (NMI
  ~0,68) y se parece mucho más a la clientela que al grupo de MCC. Con resolución 1 solo
  separa ciudades.
- **P&L:**
  - brecha de conservación de 3,8e-6 COP sobre 21,5 billones de MDR;
  - `compute_pnl` tarda 1,5 s sobre 6M filas;
  - ingreso de la red de 1.307.886.180 COP (13 pb, por construcción).

### Decisiones

- **F0 retirada** (decisión tuya): se borraron `simple_generator.py`, el notebook 00 y
  `test_vertical_slice.py`. La conservación vive en `tests/test_simple_pnl.py`.
- **Datos:**
  - 24 meses y 6M compras: dos ciclos anuales para la F8;
  - los crudos son solo hechos, sin tarifas;
  - las etiquetas latentes se escriben aparte, en `data/raw/latent/`, como verdad de terreno
    para la F4, nunca como variables.
- **Librerías:** matplotlib para los notebooks (se ven en GitHub); plotly queda para la app.
  **igraph** en lugar de networkx: Leiden en C, 3,2 MB; rustworkx no tiene detección de
  comunidades.
- **Módulo extra `population.py`**, que no estaba en el plan, para separar la generación de
  poblaciones de los contratos de `entities.py`.
- **Fraude:** calibrado contra la estructura *observada* del BCE, no contra la tasa base,
  resolviendo analíticamente con el valor esperado (probabilidad × monto). Los tests de fraude
  usan ese mismo valor esperado, porque el realizado es ruidoso en muestras chicas.
- **MDR blended por grupo** = costo del grupo (interchange + scheme fee) + ~0,45 pp.
- **Grafo:** resolución 2 de Leiden en los tests y el notebook.
- **Tests `slow` excluidos por defecto** (`addopts = -m 'not slow'`).

### Pendientes

- Contrastar la tabla de interchange con las que publican Mastercard y Redeban en Colombia
  (Decreto 1692 de 2020). Hoy devuelven 403.
- Actualizar las anclas de nivel con las cifras de 2025 del reporte de BanRep; el PDF no se
  pudo leer sin poppler ni pypdf.
- En la carpeta principal tienes un comentario sin commitear en
  `src/ips/data_gen/simple_generator.py`. Esta rama borra ese archivo: descarta o mueve la
  nota antes de hacer `git pull`.
- El `.venv` es Python 3.11.9; CLAUDE.md y la spec piden 3.12.
- `CLAUDE.md` vive en `C:\Users\USUARIO\` y no en el repo.
- Evaluar en la F3 liquidar en unidades enteras (centavos), para que la conservación y el
  determinismo sean exactos sin tolerancia.
- El `Makefile` llega con la F2.

## Feature 0 — Vertical slice (cerrada; retirada en la Feature 1)

PR #1, mergeado. Tajada vertical con generador simple, P&L por actor y test de conservación.
En la Feature 1 se retiró el generador y el notebook 00; el P&L transitorio y su test de
conservación siguen vivos sobre los datos nuevos.
