# Interchange Pricing Simulator — Especificación

> Documento de contexto del proyecto. La Parte 1 explica la economía del interchange con el detalle que un analista de pricing de una red de pagos maneja a diario. La Parte 2 traduce esa economía en el diseño del simulador. La Parte 3 define la estructura del repositorio, las features y los archivos de código de cada una.

---

# Parte 1 — Qué es el interchange

## 1.1 El modelo de cuatro partes

Cuando una persona paga con tarjeta en un comercio intervienen, como mínimo, cuatro actores además de la red:

| Rol | Quién es | Qué hace |
|---|---|---|
| **Tarjetahabiente** | La persona que paga | Usa la tarjeta emitida por su banco |
| **Emisor (issuer)** | El banco del tarjetahabiente | Emite la tarjeta, asume el riesgo de crédito y de fraude, paga recompensas |
| **Comercio (merchant)** | Quien vende | Acepta la tarjeta y recibe el dinero menos comisiones |
| **Adquirente (acquirer)** | El banco o procesador del comercio | Afilia al comercio, le entrega el dinero, le cobra la comisión |
| **Red (scheme)** | Visa, Mastercard, etc. | Fija las reglas, enruta mensajes, compensa y liquida entre emisor y adquirente |

La red **no toca el interchange**. Es lo que más se confunde: el interchange es una transferencia **del adquirente al emisor**, cuyo monto lo define la red pero que la red no se queda. La red gana con otras tarifas.

### Funciones de la red

1. **Enrutamiento de mensajes (autorización)**: es el switch que conecta miles de emisores con miles de adquirentes en milisegundos, sin que cada par tenga que integrarse bilateralmente.
2. **Compensación y liquidación**: netea multilateralmente las posiciones del día, aplica interchange y tarifas, y mueve el dinero entre bancos.
3. **Reglas del sistema**: liability shift, contracargos y disputas, honor all cards, estándares técnicos (EMV, tokenización, 3-D Secure), formatos de mensaje, plazos.
4. **Fijación del interchange**: su herramienta para balancear los dos lados del mercado.
5. **Marca y servicios de valor agregado**: scoring de fraude, tokenización, datos y analítica, transferencias (Visa Direct, Mastercard Send).

Su ingreso: assessments sobre el volumen, tarifas por autorización, cross-border y conversión de moneda, y servicios. Escala con el volumen, no con el riesgo de crédito.

### Redes de tres partes

American Express y Diners son emisor, adquirente y red a la vez. No hay interchange como transferencia entre bancos, sino un MDR único. Más control y margen por transacción, menos escala.

### Emisor y adquirente casi nunca son el mismo banco

El comercio elige su adquirente sin saber con qué banco tienen tarjeta sus clientes. Cuando sí coinciden se llama transacción **on-us**: el banco se paga el interchange a sí mismo y, según las reglas de la franquicia, puede procesarse internamente. Importa porque:

- El grado de on-us reduce el costo efectivo del adquirente grande y le permite competir en MDR.
- Un banco grande tiene incentivos contradictorios: como emisor quiere interchange alto, como adquirente lo quiere bajo.
- En el generador, el adquirente del comercio debe asignarse de forma **independiente** del emisor del tarjetahabiente, marcando las on-us con un campo aparte.

## 1.2 Anatomía de una transacción de 100.000 COP

Compra de 100.000 COP con tarjeta de crédito en un supermercado:

```
Tarjetahabiente paga ................................. 100.000
Comercio recibe ...................................... 97.500   (MDR de 2,5%)

El MDR de 2.500 se reparte así:
  Interchange → al emisor ............................ 1.700   (1,70%)
  Scheme fee / assessment → a la red ................. 130     (0,13%)
  Margen del adquirente + procesamiento .............. 670     (0,67%)
```

- **MDR (Merchant Discount Rate)**: comisión total que paga el comercio. En Colombia, "comisión de adquirencia".
- **Interchange fee**: porción del MDR que el adquirente transfiere al emisor. El componente más grande y el que fija la red.
- **Scheme fees / assessments**: lo que la red cobra a emisores y adquirentes. Es el ingreso real de Visa o Mastercard.

Flujo de fondos: el emisor debita 100.000 al tarjetahabiente, transfiere 98.300 a la red (100.000 menos interchange), la red liquida al adquirente, y el adquirente paga al comercio 97.500 tras descontar su margen y las scheme fees.

## 1.3 Por qué existe el interchange

El interchange es la herramienta con la que la red **equilibra un mercado de dos lados**. El sistema solo funciona si hay muchos tarjetahabientes y muchos comercios a la vez, y cada lado tiene una disposición distinta a pagar:

- **Lado emisor**: emitir cuesta. Pérdidas de crédito, fraude, costo de financiar el período de gracia (en crédito paga al comercio hoy y cobra al cliente en 30+ días), servicio al cliente y recompensas. Sin interchange tendría que cobrar todo eso vía cuotas de manejo e intereses, y emitiría menos.
- **Lado comercio**: aceptar tarjetas trae ventas incrementales, menos manejo de efectivo, ticket promedio más alto y acceso a clientes con crédito. Por eso tolera pagar una comisión.

La red fija el interchange donde maximiza el volumen: suficientemente alto para que los emisores tengan incentivo de emitir y promover el uso, suficientemente bajo para que los comercios no dejen de aceptar. **Ese equilibrio es el problema de pricing.**

Argumento de los emisores: el interchange financia recompensas y crédito gratuito, lo que aumenta el gasto y beneficia a los comercios. Argumento de los comercios: es un costo opaco que no pueden negociar y que pagan aunque no quieran las recompensas del cliente. Los reguladores están en el medio.

## 1.4 De qué depende la tarifa

El interchange no es un número: es una **tabla con cientos de celdas**. Dimensiones que la mueven:

**1. Tipo de producto de tarjeta** (la de mayor impacto)
- Débito < crédito estándar < crédito premium (Gold, Platinum, Signature, World Elite) < comercial/corporativa.
- Las premium traen tarjetahabientes de mayor gasto y el emisor necesita más interchange para financiar sus recompensas. Las corporativas tienen más costo de datos (nivel 2 y 3) y servicio.
- Un comercio no controla qué tarjeta le presentan: su MDR efectivo depende del **mix de tarjetas** de sus clientes.

**2. Categoría del comercio (MCC)**
Código de 4 dígitos (5411 supermercados, 5812 restaurantes, 5541 gasolineras, 4511 aerolíneas). Tarifas especiales cuando:
- Los márgenes del comercio son bajos y la aceptación es sensible al precio: supermercados, gasolineras, servicios públicos, transporte.
- Hay interés estratégico en desplazar efectivo: pagos recurrentes, transporte, micropagos.
- El riesgo de fraude o contracargo es mayor: viajes, apuestas, e-commerce de alto riesgo (tarifas más altas).

**3. Canal y nivel de seguridad**
- **Card-present (CP)** con chip/PIN o contactless: menor fraude, menor tarifa.
- **Card-not-present (CNP)**: e-commerce, teléfono. Mayor fraude, mayor tarifa, con descuentos por 3-D Secure o tokenización.
- Banda magnética o manual: la más alta.

**4. Geografía**
- **Doméstico**: emisor y comercio en el mismo país.
- **Cross-border**: tarifas notablemente más altas más comisión de conversión de moneda. Aquí la red y el emisor ganan mucho más.

**5. Tamaño del comercio y acuerdos**
Grandes comercios negocian tarifas preferenciales a cambio de volumen o exclusividad ("strategic merchant program").

**6. Monto de la transacción**
Casi todas las tarifas son **porcentual + fijo** (p. ej., 1,5% + 300 COP). En micropagos el fijo domina y el porcentaje efectivo se dispara; por eso existen tarifas especiales para transacciones pequeñas.

## 1.5 Cómo se cobra al comercio: blended vs Interchange++

- **Blended (tarifa plana)**: el comercio paga un porcentaje único sin importar la tarjeta. El adquirente absorbe la variabilidad y gana más margen en las transacciones baratas. Habitual para pymes.
- **Interchange++ (IC++)**: el comercio paga el interchange real de cada transacción + scheme fees + margen fijo del adquirente. Transparente y más barato para grandes comercios con mix favorable.

Importa para el proyecto porque **la elasticidad de aceptación depende de qué precio ve el comercio**. Un comercio en IC++ reacciona a cambios de interchange directamente; uno en blended solo si el adquirente le traslada el cambio.

## 1.6 Economía de cada actor

**Emisor** (por cada 100 unidades de gasto en crédito):
```
+ Interchange ............................ 1,5 – 2,0
+ Intereses y cuotas .................... variable, mayor
– Recompensas (cashback, millas) ......... 0,5 – 1,5 en premium
– Fraude neto ............................ 0,05 – 0,15
– Costo de fondeo del período de gracia .. 0,1 – 0,3
– Pérdidas de crédito .................... según portafolio
– Procesamiento y servicio ............... 0,1 – 0,2
```
El interchange es lo que hace rentable al cliente que paga todo cada mes (el "transactor") y no genera intereses.

**Adquirente**: gana el margen entre MDR e (interchange + scheme fees). En mercado competitivo es delgado (0,2–0,7%); compite por volumen y servicios adicionales.

**Red**: assessments a ambos lados (0,10–0,15% del volumen), tarifas por autorización, cross-border y servicios. Fija el interchange sin cobrarlo, pero **su pricing determina cuánto volumen fluye**: interchange alto → emisores promueven → más gasto; muy alto → comercios rechazan o surchargean → menos gasto. Ahí está la optimización.

**Comercio**: paga el MDR. Compara costo contra ventas incrementales y contra alternativas (efectivo, PSE, Bre-B, billeteras). En sectores de margen bajo (2–4%), un MDR de 2,5% es una fracción enorme del margen.

## 1.7 Regulación

| Jurisdicción | Medida | Efecto observado |
|---|---|---|
| **Unión Europea** (IFR, 2015) | Tope de 0,2% débito y 0,3% crédito en tarjetas de consumo | Caída del MDR, reducción de recompensas, migración a tarjetas comerciales (no topadas) |
| **Estados Unidos** (Durbin, 2011) | Tope al interchange de débito para bancos grandes (~21 centavos + 0,05%) | Bancos eliminaron débito con recompensas y subieron cuotas; crédito quedó libre |
| **Australia** (RBA, desde 2003) | Topes de promedio ponderado y permiso de surcharge | Interchange bajó de forma sostenida; se popularizó el recargo por tarjeta |
| **Colombia** | Sin tope de tarifa. SIC ha actuado por competencia (sanciones a Credibanco y Redeban en los 2000); Banco de la República regula desde 2022 los sistemas de pago de bajo valor y exige transparencia; SFC vigila emisores y adquirentes | Tarifas históricamente altas frente a Europa; nuevos adquirentes y agregadores (Bold, Wompi, PayU, Mercado Pago) presionan el MDR hacia abajo |

> ⚠️ Verificar la resolución vigente del Banco de la República sobre sistemas de pago de bajo valor antes de publicar.

La regulación es la principal fuente de **escenarios** del simulador.

## 1.8 Reglas de red que afectan la elasticidad

- **Honor all cards**: quien acepta la marca debe aceptar todas sus tarjetas (incluidas las premium más caras).
- **No-surcharge rule**: donde se permite el recargo (Australia, parte de EE. UU.), la elasticidad del comercio es más alta.
- **Steering**: descuento por efectivo o dirigir al cliente a otro medio da más poder al comercio.

## 1.9 Tendencias

- **Pagos instantáneos** (Pix, UPI, Bre-B): sustitutos de bajo costo que presionan el débito y las transacciones pequeñas.
- **Billeteras y BNPL**: a veces se apilan sobre la red (pagan interchange), a veces la evitan.
- **Tokenización y 3DS**: reducen fraude CNP y permiten tarifas diferenciadas por seguridad.
- **Tarjetas comerciales**: el segmento con más crecimiento porque suele estar fuera de los topes.

## 1.10 Glosario

| Término | Significado |
|---|---|
| **GDV** | Gross Dollar Volume: volumen bruto procesado por la red |
| **MDR** | Merchant Discount Rate: comisión total que paga el comercio |
| **Interchange** | Porción del MDR que el adquirente transfiere al emisor; la fija la red |
| **Scheme fee / assessment** | Lo que cobra la red a emisores y adquirentes |
| **MCC** | Merchant Category Code |
| **CP / CNP** | Card-present / card-not-present |
| **IC++** | Pricing transparente: interchange + scheme fee + margen |
| **Blended** | Pricing plano al comercio |
| **On-us** | Emisor y adquirente son la misma entidad |
| **Transactor / revolver** | Paga todo cada mes / financia saldo |
| **Surcharge** | Recargo por pagar con tarjeta |
| **Honor all cards** | Obligación de aceptar todas las tarjetas de una marca |
| **Cross-border** | Emisor y comercio en países distintos |
| **Net revenue yield** | Ingreso neto de la red / GDV |

---

# Parte 2 — Diseño del proyecto

## 2.1 Pregunta de negocio

> Dada una red de pagos con emisores, adquirentes, comercios y tarjetahabientes, ¿cómo debería fijarse la tabla de interchange por segmento de comercio y producto de tarjeta para maximizar el ingreso neto de la red, sujeto a restricciones de aceptación, rentabilidad mínima de los emisores y topes regulatorios?

| Área | Pregunta del proyecto |
|---|---|
| Profitability assessments | ¿Qué segmentos y productos son rentables para red, emisor y adquirente? |
| Customer and product segmentation | ¿Cómo se agrupan los comercios según su estructura de clientes y no solo su MCC? |
| Forecasting | ¿Cuánto volumen tendrá cada segmento en 12 meses? |
| Simulations | ¿Qué pasa si se topa el interchange, entra un pago instantáneo, o un gran comercio negocia? |
| Optimization | ¿Cuál es la tabla de interchange óptima bajo restricciones? |
| Revenue analytics | Descomposición del ingreso por driver: volumen, mix, tarifa, cross-border |

## 2.2 Modelo de datos

```
issuer            (issuer_id, país, tipo: banco/fintech, tamaño, mix de productos)
acquirer          (acquirer_id, país, modelo de pricing: blended / IC++)
merchant          (merchant_id, mcc, país, ciudad, tamaño, acquirer_id, canal: CP/CNP/mixto,
                   margen del sector, estado de aceptación por producto)
cardholder        (cardholder_id, issuer_id, país, producto: débito/crédito std/premium/comercial,
                   segmento de gasto, propensión cross-border)
transaction       (txn_id, fecha, cardholder_id, merchant_id, monto, canal, cross_border, on_us,
                   interchange_aplicado, scheme_fee, mdr, fraude_flag, contracargo_flag)
interchange_table (producto, mcc_group, canal, región, pct, fijo, vigencia)
```

**Grafo** (módulo de Graph ML): nodos tarjetahabientes y comercios (opcional: emisores y adquirentes como segundo nivel). Aristas tarjetahabiente → comercio ponderadas por número de transacciones y monto en una ventana. Bipartito, disperso, distribución de grado de cola larga.

## 2.3 Generador de datos sintéticos

No existen datos reales de interchange públicos. El generador produce datos con estructura creíble y documenta cada supuesto:

1. **Comercios**: distribución de MCC calibrada al comercio colombiano (retail y alimentos dominan). Tamaño log-normal. Margen del sector como parámetro por MCC.
2. **Tarjetahabientes**: mix de productos según emisor (bancos grandes más premium; fintechs más débito). Gasto mensual log-normal por segmento.
3. **Transacciones**: proceso de afinidad: cada tarjetahabiente tiene preferencias latentes por categorías y geografía, lo que produce comunidades naturales en el grafo. Estacionalidad mensual y semanal. Cross-border como fracción baja concentrada en premium.
4. **Tarifas**: tabla inicial inspirada en tablas públicas (0,3–0,5% débito en categorías reducidas, 1,2–2,2% crédito, hasta 2,5–3% premium CNP), con componente fijo.
5. **Fraude y contracargos**: probabilidades por canal y MCC.

Salida: 5–20 millones de transacciones, para que el módulo de datos sea un problema real de ingeniería.

## 2.4 Módulo de rentabilidad (unit economics)

```
Ingreso red        = scheme_fee + tarifa_cross_border + tarifa_por_autorización
Ingreso emisor     = interchange – recompensas – fraude_neto – costo_fondeo – pérdida_crédito_esperada
Ingreso adquirente = mdr – interchange – scheme_fee – costo_procesamiento
Costo comercio     = mdr  (vs margen del sector → "carga relativa")
```

Métricas clave: **net revenue yield** de la red (ingreso neto / GDV), **interchange efectivo** por segmento, **carga relativa del comercio** (MDR / margen sectorial; mejor predictor de abandono de aceptación), **contribución del emisor por producto**.

## 2.5 Segmentación con Graph ML

**Baseline**: k-means sobre atributos (MCC, tamaño, ticket promedio, canal, mix de tarjetas). Es lo que hace la industria hoy.

**Propuesta**: embeddings de comercios sobre el grafo bipartito (node2vec o GraphSAGE) + clustering. Dos comercios quedan cerca si comparten tarjetahabientes aunque tengan MCC distinto. Captura:
- Sustitutos (compiten por el mismo cliente): a dónde migra el gasto si uno deja de aceptar.
- Complementarios (cliente que va a A también va a B): programas de comercio estratégico.
- Comercios cuyo cliente real es distinto de lo que su MCC sugiere.

**Evaluación honesta**: comparar por (a) homogeneidad del interchange efectivo dentro del clúster, (b) capacidad de predecir abandono de aceptación en el simulador, (c) interpretabilidad. Si el grafo no gana en (b), se reporta.

**Link prediction**: predice aristas tarjetahabiente–comercio → estima volumen que un comercio ganaría/perdería al cambiar aceptación → alimenta la elasticidad.

**Comunidades**: Leiden sobre el grafo proyectado de comercios.

## 2.6 Elasticidad y simulación

- **Elasticidad de aceptación del comercio**: probabilidad de dejar de aceptar un producto en función de carga relativa, canal, alternativas (pago instantáneo, efectivo) y posibilidad de surcharge. Curva logística parametrizada, calibrada con literatura y documentada como supuesto.
- **Elasticidad de uso del tarjetahabiente**: cambio en gasto con tarjeta cuando el emisor reduce recompensas.

**Motor**: dado un cambio en la tabla, recalcula en cadena:
1. Nuevo MDR por comercio (blended o IC++).
2. Nueva carga relativa → probabilidad de abandono → volumen perdido (redistribuido con link prediction).
3. Nuevo ingreso del emisor → ajuste de recompensas → cambio en gasto del tarjetahabiente.
4. P&L de todos los actores.

**Escenarios estándar**:
- Tope regulatorio al débito (0,3%) y al crédito (0,5%).
- Pago instantáneo que sustituye 20% de las transacciones de débito bajo 50.000 COP.
- Gran comercio negocia tarifa preferencial.
- Migración de 10% del volumen de crédito estándar a premium.
- Descuento de interchange en CNP tokenizado.

## 2.7 Optimización

- **Variables**: tarifa porcentual y fija por celda (producto × grupo de MCC × canal).
- **Objetivo**: maximizar ingreso neto de la red a 12 meses.
- **Restricciones**: aceptación mínima por segmento; rentabilidad mínima por producto para el emisor mediano; topes regulatorios; suavidad (cambio máximo por celda por año).
- **Método**: búsqueda bayesiana o evolutiva sobre el simulador (no lineal y estocástico). Reportar frente de Pareto ingreso vs aceptación.

## 2.8 Forecasting

Volumen mensual por segmento con modelo jerárquico (segmento → MCC → comercio) usando Prophet o modelo aditivo con estacionalidad, más ajuste por cambios estructurales del simulador. Mostrar que forecast y simulador comparten segmentación y supuestos.

## 2.9 Entregables

1. **Repositorio** con generador, pipeline (dbt sobre DuckDB), grafo, simulador y optimizador. README en inglés con diagrama.
2. **Dashboard** (Streamlit): P&L por actor y segmento, explorador de escenarios, frente de Pareto.
3. **Memo ejecutivo** de dos páginas con tres recomendaciones de pricing. Para un VP de Pricing, sin jerga.
4. **Nota metodológica**: supuestos, calibración, límites del dato sintético, grafo vs baseline.

---

# Parte 3 — Estructura del repositorio y features

## 3.1 Principio de ejecución

Primero una **tajada vertical mínima que produzca un número de negocio**, luego profundizar cada capa. Orden sugerido: `0 → 1 → 2 → 3 → 4 → 5 → 6 → 9 (borrador) → 8 → 7 → 9 (final)`. Camino crítico: 1-2-3-5-6. Paralelizables: 4 y 8. Primera en recortar si falta tiempo: 7.

Una rama por feature, un directorio por módulo, issues de GitHub por feature.

## 3.2 Árbol del repositorio

```
interchange-pricing-simulator/
├── README.md
├── CLAUDE.md
├── pyproject.toml
├── Makefile
├── .github/workflows/ci.yml
├── config/
│   ├── base.yaml                 # parámetros globales (semilla, tamaño, fechas)
│   ├── interchange_table.yaml    # tabla inicial producto × mcc_group × canal × región
│   ├── mcc_groups.yaml           # mapeo MCC → grupo y margen sectorial
│   └── scenarios/
│       ├── regulatory_cap_debit.yaml
│       ├── regulatory_cap_credit.yaml
│       ├── instant_payments_entry.yaml
│       ├── strategic_merchant_deal.yaml
│       ├── premium_migration.yaml
│       └── cnp_tokenized_discount.yaml
├── data/
│   ├── raw/                      # salida del generador (parquet)
│   ├── warehouse/                # duckdb file
│   └── artifacts/                # embeddings, modelos, resultados
├── src/ips/                      # paquete Python "ips"
│   ├── __init__.py
│   ├── data_gen/
│   ├── economics/
│   ├── graph/
│   ├── elasticity/
│   ├── simulator/
│   ├── optimizer/
│   ├── forecast/
│   └── utils/
│       ├── io.py                 # lectura/escritura parquet, conexión duckdb
│       ├── config.py             # carga y validación de YAML (pydantic)
│       └── logging.py
├── dbt/
│   ├── dbt_project.yml
│   ├── profiles.yml
│   ├── models/
│   │   ├── staging/
│   │   ├── core/
│   │   └── marts/
│   ├── tests/
│   └── macros/
├── notebooks/
│   ├── 00_vertical_slice.ipynb
│   ├── 01_data_validation.ipynb
│   ├── 02_unit_economics.ipynb
│   ├── 03_segmentation_comparison.ipynb
│   ├── 04_elasticity_calibration.ipynb
│   ├── 05_scenarios.ipynb
│   └── 06_pareto_front.ipynb
├── app/
│   ├── streamlit_app.py
│   ├── pages/
│   └── components/
├── reports/
│   ├── executive_memo.md
│   ├── methodology_note.md
│   └── figures/
├── tests/
└── docs/
    ├── spec.md
    ├── progress.md
    ├── architecture.md
    ├── assumptions.md
    └── architecture_diagram.png
```

## 3.3 Feature 0 — Vertical slice

**Objetivo**: generador simplista (1 tabla, 3 productos, 10 MCC), pipeline mínimo, P&L por actor, un notebook que imprime "ingreso de la red = X".

| Archivo | Qué contiene |
|---|---|
| `src/ips/data_gen/simple_generator.py` | Genera issuers, merchants, cardholders y transactions con distribuciones uniformes/normales; ~100k txns |
| `src/ips/economics/simple_pnl.py` | Función `compute_pnl(df, table)` que devuelve ingreso por actor |
| `notebooks/00_vertical_slice.ipynb` | Corre todo end-to-end y muestra el número |
| `config/base.yaml` | Semilla, tamaños, fechas |
| `tests/test_vertical_slice.py` | Verifica que el P&L cuadra: suma de ingresos = MDR total |

**Criterio de terminado**: el notebook corre en < 1 min y el test de conservación pasa.

## 3.4 Feature 1 — Generador de datos sintéticos

**Objetivo**: 5M+ transacciones con estructura creíble y supuestos documentados.

| Archivo | Qué contiene |
|---|---|
| `src/ips/data_gen/entities.py` | Clases `Issuer`, `Acquirer`, `Merchant`, `Cardholder` (pydantic) |
| `src/ips/data_gen/distributions.py` | Distribuciones parametrizadas: MCC mix colombiano, tamaño log-normal, mix de producto por tipo de emisor |
| `src/ips/data_gen/affinity.py` | Preferencias latentes por categoría y geografía; genera la matriz de afinidad tarjetahabiente–comercio que produce comunidades |
| `src/ips/data_gen/transactions.py` | Proceso de transacciones con estacionalidad semanal/mensual, cross-border, canal CP/CNP, marca on-us |
| `src/ips/data_gen/interchange_table.py` | Construye la tabla completa desde `config/interchange_table.yaml`; función `lookup(product, mcc_group, channel, region, amount)` |
| `src/ips/data_gen/fraud.py` | Probabilidades de fraude y contracargo por canal y MCC |
| `src/ips/data_gen/generate.py` | CLI `python -m ips.data_gen.generate --config config/base.yaml` que orquesta todo y escribe parquet a `data/raw/` |
| `config/mcc_groups.yaml` | MCC → grupo, margen sectorial, elasticidad base |
| `docs/assumptions.md` | Cada distribución y su justificación |
| `tests/test_data_gen.py` | Distribuciones dentro de rangos esperados; el grafo tiene comunidades detectables |
| `notebooks/01_data_validation.ipynb` | Histogramas y sanity checks |

**Criterio de terminado**: 5M+ txns, tests pasan, `assumptions.md` completo.

## 3.5 Feature 2 — Pipeline y datasets analíticos

**Objetivo**: capa dbt reutilizable (migrable a Snowflake cambiando solo `profiles.yml`).

| Archivo | Qué contiene |
|---|---|
| `dbt/models/staging/stg_transactions.sql` | Tipado y limpieza de `data/raw/transactions.parquet` |
| `dbt/models/staging/stg_merchants.sql`, `stg_cardholders.sql`, `stg_issuers.sql`, `stg_acquirers.sql` | Staging por entidad |
| `dbt/models/staging/schema.yml` | Tests `not_null`, `unique`, `accepted_values` |
| `dbt/models/core/fct_transactions.sql` | Hechos con interchange, scheme fee y MDR calculados por fila |
| `dbt/models/core/dim_merchant.sql`, `dim_cardholder.sql`, `dim_issuer.sql`, `dim_acquirer.sql`, `dim_date.sql` | Dimensiones |
| `dbt/models/marts/mart_pnl_by_actor.sql` | P&L agregado por actor, mes, producto, mcc_group |
| `dbt/models/marts/mart_effective_interchange.sql` | Interchange efectivo por segmento |
| `dbt/models/marts/mart_merchant_relative_burden.sql` | MDR / margen sectorial por comercio |
| `dbt/models/marts/mart_graph_edges.sql` | Aristas cardholder–merchant agregadas por ventana (peso = n_txns, monto) |
| `dbt/macros/interchange_lookup.sql` | Macro que aplica la tabla de interchange en SQL |
| `dbt/tests/assert_pnl_conservation.sql` | Test singular: suma de ingresos por actor = MDR total |
| `src/ips/utils/io.py` | Conexión DuckDB, helpers de parquet |
| `Makefile` | `make generate`, `make dbt`, `make test` |
| `.github/workflows/ci.yml` | Lint + tests + `dbt build` sobre muestra pequeña |

**Criterio de terminado**: `dbt build` verde, lineage documentado en dbt docs.

## 3.6 Feature 3 — Unit economics y revenue analytics

**Objetivo**: P&L por actor y descomposición del ingreso por driver.

| Archivo | Qué contiene |
|---|---|
| `src/ips/economics/pnl.py` | `IssuerPnL`, `AcquirerPnL`, `NetworkPnL`, `MerchantCost` con parámetros (recompensas, fondeo, fraude, procesamiento) |
| `src/ips/economics/revenue_bridge.py` | Descomposición volumen / mix / tarifa / cross-border entre dos períodos o escenarios |
| `src/ips/economics/metrics.py` | `net_revenue_yield`, `effective_interchange`, `relative_burden`, `issuer_contribution` |
| `src/ips/economics/params.py` | Parámetros económicos por producto y actor (pydantic) |
| `tests/test_economics.py` | Conservación y casos de borde (micropagos, cross-border) |
| `notebooks/02_unit_economics.ipynb` | Primer dashboard estático con gráficos |

## 3.7 Feature 4 — Segmentación (Graph ML)

**Objetivo**: baseline vs grafo con evaluación de negocio.

| Archivo | Qué contiene |
|---|---|
| `src/ips/graph/build_graph.py` | Construye grafo bipartito desde `mart_graph_edges` (networkx / PyG `HeteroData`) |
| `src/ips/graph/baseline_kmeans.py` | Features tabulares + k-means; selección de k |
| `src/ips/graph/node2vec_embed.py` | Embeddings node2vec (PyG) con parámetros p, q, dims |
| `src/ips/graph/graphsage_embed.py` | GraphSAGE no supervisado; solo si node2vec no basta |
| `src/ips/graph/clustering.py` | Clustering sobre embeddings (k-means, HDBSCAN) |
| `src/ips/graph/communities.py` | Proyección a grafo de comercios + Leiden |
| `src/ips/graph/evaluate.py` | Métricas: homogeneidad de interchange efectivo, poder predictivo de abandono, silhouette, interpretabilidad (top MCC por clúster) |
| `src/ips/graph/segment_profiles.py` | Perfil de cada segmento para el memo |
| `data/artifacts/embeddings/` | Embeddings versionados |
| `tests/test_graph.py` | Grafo bien formado; embeddings reproducibles con semilla |
| `notebooks/03_segmentation_comparison.ipynb` | Comparación honesta baseline vs grafo |

**Criterio de terminado**: tabla comparativa con las tres métricas y conclusión explícita.

## 3.8 Feature 5 — Elasticidad

**Objetivo**: curvas de respuesta documentadas + link prediction.

| Archivo | Qué contiene |
|---|---|
| `src/ips/elasticity/merchant_acceptance.py` | Curva logística P(abandono) según carga relativa, canal, alternativas y surcharge permitido; parámetros por segmento |
| `src/ips/elasticity/cardholder_response.py` | Cambio de gasto ante cambio en recompensas |
| `src/ips/elasticity/link_prediction.py` | Modelo de link prediction (PyG) para redistribuir volumen perdido |
| `src/ips/elasticity/calibration.py` | Calibración de parámetros con datos sintéticos y referencias de literatura |
| `src/ips/elasticity/params.py` | Parámetros de elasticidad (pydantic) |
| `docs/assumptions.md` (sección elasticidad) | Cada supuesto y su fuente |
| `tests/test_elasticity.py` | Monotonicidad de las curvas; link prediction supera baseline aleatorio |
| `notebooks/04_elasticity_calibration.ipynb` | Curvas visualizadas por segmento |

## 3.9 Feature 6 — Simulador de escenarios

**Objetivo**: motor que recalcula la cadena completa y corre escenarios desde YAML.

| Archivo | Qué contiene |
|---|---|
| `src/ips/simulator/engine.py` | Clase `Simulator` con `run(scenario) -> SimulationResult`; encadena tarifa → MDR → abandono → recompensas → gasto → P&L |
| `src/ips/simulator/scenario.py` | Carga y validación de `config/scenarios/*.yaml` (pydantic) |
| `src/ips/simulator/state.py` | Estado de la red (tabla vigente, aceptación por comercio, recompensas por emisor) |
| `src/ips/simulator/result.py` | `SimulationResult`: P&L por actor, aceptación, volumen, comparación vs baseline |
| `src/ips/simulator/sensitivity.py` | Barridos de un parámetro y tornado charts |
| `src/ips/simulator/cli.py` | `python -m ips.simulator run --scenario config/scenarios/regulatory_cap_debit.yaml` |
| `config/scenarios/*.yaml` | Los seis escenarios estándar |
| `tests/test_simulator.py` | Escenario nulo = baseline; tope reduce ingreso del emisor; determinismo con semilla |
| `notebooks/05_scenarios.ipynb` | Resultados de los seis escenarios |

**Criterio de terminado**: los seis escenarios corren en < 5 min y sus resultados son explicables en una frase cada uno.

## 3.10 Feature 7 — Optimización

**Objetivo**: tabla de interchange óptima bajo restricciones y frente de Pareto.

| Archivo | Qué contiene |
|---|---|
| `src/ips/optimizer/problem.py` | Variables (celdas), objetivo, restricciones (aceptación mínima, rentabilidad emisor, topes, suavidad) |
| `src/ips/optimizer/bayesian.py` | Optuna sobre el simulador |
| `src/ips/optimizer/evolutionary.py` | Alternativa NSGA-II (pymoo) para multiobjetivo |
| `src/ips/optimizer/pareto.py` | Construcción y visualización del frente ingreso vs aceptación |
| `src/ips/optimizer/cli.py` | `python -m ips.optimizer run --budget 500` |
| `tests/test_optimizer.py` | Respeta restricciones; mejora sobre tabla inicial |
| `notebooks/06_pareto_front.ipynb` | Frente de Pareto y tabla recomendada |

## 3.11 Feature 8 — Forecasting

**Objetivo**: volumen mensual por segmento con modelo jerárquico.

| Archivo | Qué contiene |
|---|---|
| `src/ips/forecast/hierarchical.py` | Prophet o modelo aditivo por segmento → MCC → comercio con reconciliación |
| `src/ips/forecast/adjust_structural.py` | Aplica cambios estructurales del simulador al forecast |
| `src/ips/forecast/evaluate.py` | Backtesting, MAPE por nivel |
| `tests/test_forecast.py` | Reconciliación suma correctamente |

## 3.12 Feature 9 — Capa de negocio

**Objetivo**: dashboard, memo y documentación final.

| Archivo | Qué contiene |
|---|---|
| `app/streamlit_app.py` | Entrada de la app |
| `app/pages/1_pnl_overview.py` | P&L por actor y segmento |
| `app/pages/2_segments.py` | Explorador de segmentos y embeddings |
| `app/pages/3_scenario_explorer.py` | Controles de tarifa por celda y resultado en vivo |
| `app/pages/4_pareto.py` | Frente de Pareto interactivo |
| `app/components/charts.py` | Gráficos reutilizables (plotly) |
| `reports/executive_memo.md` | Dos páginas, tres recomendaciones, sin jerga; borrador tras Feature 6 |
| `reports/methodology_note.md` | Supuestos, calibración, límites del dato sintético, grafo vs baseline |
| `docs/architecture.md` + `architecture_diagram.png` | Diagrama para README y LinkedIn |
| `README.md` | En inglés: problema, arquitectura, cómo correrlo, resultados clave |

## 3.13 Stack

| Capa | Herramienta |
|---|---|
| Datos | Python 3.12, polars, pyarrow, DuckDB |
| Transformación | dbt-duckdb |
| Grafo | PyTorch Geometric, networkx, igraph/leidenalg |
| Optimización | Optuna, pymoo, scipy |
| Forecast | Prophet o statsforecast |
| App | Streamlit + plotly |
| Calidad | pytest, ruff, dbt tests, GitHub Actions |
| Config | pydantic + YAML |

Todo el stack es gratuito y corre local.
