# Estado del proyecto

## Feature en curso

Ninguna. La Feature 0 (vertical slice) está cerrada en la rama `feat/00-vertical-slice`, con PR
contra `main`. Siguiente según el orden de CLAUDE.md: **Feature 1 — Generador de datos
sintéticos**.

## Feature 0 — Vertical slice (cerrada)

### Qué se hizo

- `pyproject.toml`: paquete `ips` (layout src), dependencias y extra `dev`; configuración de
  pytest y ruff.
- `config/base.yaml` + `src/ips/utils/config.py`: todos los parámetros (semilla, tamaños, fechas,
  entidades, distribuciones, pricing) en YAML, validados con pydantic: modelos inmutables,
  `extra="forbid"`, participaciones que suman 1 y grilla de interchange completa.
- `src/ips/data_gen/simple_generator.py`: 6 emisores, 4 adquirentes, 2.000 comercios, 20.000
  tarjetahabientes y 100.000 transacciones. CLI `python -m ips.data_gen.simple_generator`.
- `src/ips/economics/simple_pnl.py`: `compute_pnl(transactions, table, pricing)`, `PnLResult`
  (P&L por actor y totales) y `pnl_by(result, keys)`.
- `tests/test_vertical_slice.py`: 19 tests.
  - Conservación del total contra un MDR calculado de forma independiente, y en 8 agregaciones.
  - P&L por actor contra el nivel fila, sin fan-out, y error ante una celda faltante o duplicada.
  - Ejemplo de spec §1.2 calculado a mano.
  - Determinismo con semilla fija e independencia adquirente ⟂ emisor.
  - Forma de las tablas y notebook en menos de 60 s.
- `notebooks/00_vertical_slice.ipynb`: corre de punta a punta en ~2 s, arranque del kernel incluido.
- `docs/assumptions.md`: supuestos F0-01 a F0-17, cada uno marcado [spec/literatura] o
  [supuesto propio].

### Resultado (datos sintéticos, semilla 20260910; valida la tubería, no es un hallazgo)

- Ingreso de la red: 25.537.989 COP sobre un GDV de 19.644.606.823 COP. Son 13,0 pb, por
  construcción.
- Reparto del MDR: emisores 63,8 %, adquirentes 29,8 %, red 6,4 %.
- On-us: 16,0 % de las transacciones.
- Margen del adquirente sobre GDV: débito 0,99 %, crédito estándar 0,16 %, premium −0,25 %.

### Decisiones

- Monto log-normal por MCC en lugar de normal. Es una desviación deliberada de spec §3.3 (F0-09).
- MDR blended para todos los comercios, con el adquirente como residuo. IC++ queda para la
  Feature 3.
- Las tasas blended quedaron entre 0,1 y 0,5 pp por debajo de las del plan, para que el margen
  promedio del adquirente por MCC quede en 0,52–0,65 %, dentro del 0,2–0,7 % de spec §1.6.
- On-us sin neteo por grupo financiero (F0-04).
- La tabla cruda de transacciones no trae tarifas: `compute_pnl` las deriva, así que los mismos
  hechos se pueden re-tarificar con otra tabla. Es la base del simulador.
- Los parámetros de la Feature 0 viven en `config/base.yaml`. En la Feature 1, la tabla de
  interchange y los MCC migran a `config/interchange_table.yaml` y `config/mcc_groups.yaml`.
- Un stream RNG por entidad (`SeedSequence.spawn`): cambiar `n_transactions` no altera
  comercios ni tarjetahabientes.
- Determinismo: con la misma semilla, los datos generados son idénticos bit a bit. Los totales
  en float se comparan con tolerancia relativa 1e-12, porque polars paraleliza las sumas y el
  orden de acumulación puede cambiar el último bit entre corridas.

### Pendientes

- El `.venv` es Python 3.11.9, pero CLAUDE.md y la spec piden 3.12. El código es compatible con
  ambos (`requires-python >= 3.11`).
- La instalación editable (`pip install -e ".[dev]"`) quedó apuntando al worktree de esta
  feature. Después del merge hay que correrla de nuevo desde el checkout principal.
- En el checkout principal, `docs/spec.md` y `docs/progress.md` están sin rastrear. Hay que
  moverlos o borrarlos antes de `git pull`, porque ahora vienen en el repo.
- `CLAUDE.md` vive en `C:\Users\USUARIO\` y no en el repo, aunque spec §3.2 lo ubica en la raíz.
- Evaluar en la Feature 3 liquidar los montos en unidades enteras (centavos), para que la
  conservación y el determinismo sean exactos sin tolerancia.
- Todavía no hay `Makefile`; llega con la Feature 2.
