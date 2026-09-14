# Supuestos del generador y del modelo económico

Los datos del proyecto son **sintéticos**: no existen datos públicos reales de interchange.
Cada supuesto se registra aquí en el momento de introducirlo, con su origen:

- **[D] dato público**: cifra oficial usada directamente (p. ej., población DANE).
- **[L] literatura**: tomado de un informe o publicación citada abajo.
- **[S] spec**: tomado de `docs/spec.md`, que resume práctica de la industria y regulación.
- **[P] supuesto propio**: elección de diseño sin calibración externa. Debe leerse como
  ilustrativa, nunca como hallazgo empírico.

Los valores exactos viven en `config/base.yaml`, `config/mcc_groups.yaml`,
`config/interchange_table.yaml`, `config/economics.yaml` y `config/elasticity.yaml`; este
documento explica el porqué.

---

## Feature 1 — Generador de datos sintéticos (vigente)

### Escala y horizonte

| # | Supuesto | Origen | Dónde |
|---|---|---|---|
| F1-01 | 24 meses (2024-01 a 2025-12), 6M compras y 150.000 tarjetas: 20 compras por tarjeta al año. Dos ciclos anuales para que el forecasting (F8) vea estacionalidad | Frecuencia [L: Superfinanciera 2022, 19–26 compras por tarjeta al año]; horizonte [P] | `base.yaml: dates, sizes` |
| F1-02 | Cardholder = tarjeta: una persona con dos tarjetas aparece dos veces | [P] | `population.py` |

### Emisores, productos y segmentos

| # | Supuesto | Origen | Dónde |
|---|---|---|---|
| F1-03 | 10 emisores ficticios: 3 bancos grandes, 3 medianos, 2 pequeños y 2 fintech | [P] | `base.yaml: issuers` |
| F1-04 | Mix de productos por tipo y tamaño de emisor, calibrado a ~74 % de tarjetas débito; el crédito se reparte 16 / 7 / 3 % entre estándar, premium y comercial | 74 % [L: Superfinanciera 2022, 45,8M débito vs 16M crédito]; reparto [P] | `base.yaml: product_mix` |
| F1-05 | Segmento de gasto (bajo, medio, alto) según producto; actividad ~ LogNormal(mediana del segmento, 0,6). El crédito resulta ~1,34x más activo y el débito ≈ 68 % de las compras | 68 % [L: Superfinanciera 2022, 878M compras débito vs 416M crédito]; forma [S §2.3] | `base.yaml: spend_segments` |
| F1-06 | Seis segmentos latentes de estilo de vida (hogar, joven urbano, viajero, empresarial, conductor, estudiante) con preferencias por grupo de MCC; la preferencia de cada tarjeta ~ Dirichlet(15 × prototipo del segmento) | [P] | `base.yaml: lifestyle`, `affinity.py` |
| F1-07 | Propensión cross-border por producto (0,5 / 2 / 6 / 5 % de las compras), duplicada en el segmento viajero | [P]; dirección [S §2.3: fracción baja concentrada en premium] | `base.yaml: cross_border` |

### Geografía

| # | Supuesto | Origen | Dónde |
|---|---|---|---|
| F1-08 | Siete áreas urbanas ponderadas por población (Bogotá+Soacha 8,80M, Medellín+Bello 3,23M, Cali 2,30M, Barranquilla+Soledad 2,02M, Cartagena 1,07M, Cúcuta 0,80M, Bucaramanga 0,62M) | [D: DANE, proyección 2023] | `base.yaml: geography` |
| F1-09 | "Otras ciudades" con 35 % de tarjetas y comercios (menor penetración); 92 % de las compras presenciales en la ciudad de residencia | [P] | `base.yaml: geography` |

### Comercios

| # | Supuesto | Origen | Dónde |
|---|---|---|---|
| F1-10 | 33 MCC en 12 grupos; retail y alimentos dominan el mix de comercios; bares y restaurantes (5812 + 5813) = 5,3 % | Dominio de retail y alimentos [S §2.3]; 5,3 % [L: Credibanco vía Asobares, 8.722 de 164.239 comercios]; resto [P] | `mcc_groups.yaml: mccs` |
| F1-11 | Tamaño del comercio ~ LogNormal(0, 1,2); tramos por cuantil: 1 % grande, 9 % mediano, 30 % pequeño, 60 % micro | Forma [S §2.3]; σ y cortes [P] | `base.yaml: merchants` |
| F1-12 | El adquirente se asigna según el tramo de tamaño (los bancos dominan los grandes, los agregadores los micro), sin mirar el emisor de los clientes; on-us = mismo grupo financiero | Independencia [S §1.1]; participaciones [P] | `base.yaml: acquirers` |
| F1-13 | Margen sectorial por grupo (3 % supermercados y gasolina … 20 % digital) × LogNormal(0, 0,25) por comercio | Ancla de sectores de margen bajo [S §1.6: 2–4 %]; resto [P] | `mcc_groups.yaml: groups` |
| F1-14 | Elasticidad base por grupo: marcador de posición, no se usa hasta la Feature 5 | [P] | `mcc_groups.yaml: groups` |
| F1-15 | Canal de cada comercio según su MCC (solo presente, solo online o mixto); 600 comercios extranjeros (digital 50 %, viajes 30 %, retail 20 %; EE. UU., España, México, Panamá, Brasil) con un adquirente extranjero que nunca es on-us | [P] | `mcc_groups.yaml`, `base.yaml: cross_border` |
| F1-16 | Clientela latente de cada comercio, proporcional a la demanda de cada segmento en su grupo: comercios de MCC distintos comparten clientela | [P] | `affinity.py` |

**Advertencia metodológica sobre F1-06 y F1-16.** Las comunidades del grafo las plantamos
nosotros. Que Leiden las recupere (notebook 01) solo prueba que el generador funciona. La
Feature 4 tiene que mostrar que detectarlas mejora métricas de negocio frente a un k-means
sobre atributos observables; si no lo logra, se reporta.

### Transacciones

| # | Supuesto | Origen | Dónde |
|---|---|---|---|
| F1-17 | Compras por tarjeta ~ Multinomial(6M, actividad) | [P] | `transactions.py` |
| F1-18 | Grupo de cada compra ~ preferencias de la tarjeta; online con la probabilidad del grupo (0 % gasolina … 100 % digital); en el exterior, 85 % online | [P] | `mcc_groups.yaml: online_share` |
| F1-19 | Hábito: 60 % de las compras van a uno de los 3 comercios habituales de la tarjeta en ese grupo (pesos 0,6 / 0,3 / 0,1); el resto explora con peso tamaño × (1 + 4 si la clientela coincide con el segmento) | [P] | `base.yaml: affinity` |
| F1-20 | Fecha: factor mensual (diciembre 1,30, junio 1,04 por la prima), tendencia de +19 % anual, quincena ×1,15 (días 1, 14–16, 29–31) y perfil semanal por grupo | Tendencia [L: BanRep 2026, +19 % en número de operaciones con tarjeta en 2025]; resto [P] | `base.yaml: seasonality`, `mcc_groups.yaml: weekday_profiles` |
| F1-21 | Monto ~ LogNormal(mediana del MCC × multiplicador de producto 0,75 / 1,0 / 1,35 / 1,65, σ del MCC), piso de 1.000 COP. Ticket promedio resultante ≈ 129k débito y 248k crédito | Objetivo 131k y 244k [L: Superfinanciera 2022, 114k y 212k]; ajuste de ~15 % por inflación 2023–2024 [P] | `mcc_groups.yaml: mccs`, `base.yaml: amounts` |
| F1-22 | 50 % de las compras no presentes son tokenizadas | [P] | `base.yaml: channel` |

### Fraude y contracargos

| # | Supuesto | Origen | Dónde |
|---|---|---|---|
| F1-23 | p(fraude) = tasa base del canal × riesgo del grupo × multiplicador cross-border × 0,5 si está tokenizada. Parámetros resueltos analíticamente para reproducir la estructura del BCE (el no presente concentra ~84 % del valor defraudado; el cross-border defrauda ~14x lo doméstico) y un total de 0,08 % del valor: 0,0174 % presente, 0,099 % no presente, ×6,4 cross-border | Estructura [L: BCE 2023]; nivel [S §1.6: 0,05–0,15 %]; riesgo por grupo [P] | `base.yaml: fraud`, `mcc_groups.yaml: fraud_risk` |
| F1-24 | Contracargo en 80 % de los fraudes y en 0,02–0,30 % de las compras legítimas según el grupo (más alto en viajes y digital) | [P] | `base.yaml: fraud`, `mcc_groups.yaml: dispute_rate` |

### Tabla de interchange

| # | Supuesto | Origen | Dónde |
|---|---|---|---|
| F1-25 | Grilla producto × grupo × canal × región (192 celdas). Base presente doméstica: débito 0,3–0,45 % en categorías reducidas y hasta 1,1 % en el resto; crédito estándar 1,0–2,0 %; premium hasta 2,5 %; comercial ≥ premium; fijo de 0–200 COP. Recargos: +0,30 pp no presente, +0,80 pp cross-border | Rangos [S §2.3]; cada celda [P]. Pendiente: contrastar con las tablas que Mastercard y Redeban publican en Colombia por el Decreto 1692 de 2020 (no se pudieron descargar: HTTP 403) | `interchange_table.yaml` |
| F1-26 | Tramo de micropagos: ≤ 20.000 COP a 0,20 % sin fijo, para débito y crédito estándar en alimentos, transporte y restaurantes, solo doméstico | [S §1.4: tarifas especiales para montos pequeños]; valores [P] | `interchange_table.yaml: small_ticket` |
| F1-27 | En cada celda: débito < crédito estándar < premium ≤ comercial | [S §1.4] | verificado en `tests/test_interchange_table.py` |

### P&L transitorio (`simple_pnl`, reemplazado en la Feature 3)

Se conserva como registro: la Feature 3 retiró `simple_pnl.py`. F1-28 sigue vivo dentro de
F3-01, F1-29 quedó reemplazado por F3-03 y F3-06, y F1-30 lo resuelve F3-05.

| # | Supuesto | Origen | Dónde |
|---|---|---|---|
| F1-28 | Scheme fee de 0,13 % del monto, cobrado al adquirente desde el MDR | [S §1.2]; rango 0,10–0,15 % [S §1.6] | Reemplazado por F3-01 |
| F1-29 | Todo comercio paga el MDR blended de su grupo, calibrado como costo del grupo + ~0,45 pp: margen del adquirente de 0,44–0,51 % por grupo | Rango de margen [S §1.6: 0,2–0,7 %]; tasas [P] | Reemplazado por F3-03 y F3-06 |
| F1-30 | En compras cross-border el comercio extranjero también paga la tarifa blended de su grupo, y el adquirente extranjero puede perder | [P] | Reemplazado por F3-05 |

### Consecuencias a tener presentes

- Hasta la Feature 2, con un scheme fee puramente porcentual, el net revenue yield de la red
  era de 13 pb por construcción. Desde la Feature 3 resulta de las tarifas de F3-01 y F3-02.
- Que el generador reproduzca sus anclas (tabla de calibración del notebook 01) prueba que es
  consistente con sus supuestos, no que describa el mercado colombiano real.
- La calibración de nivel usa Superfinanciera 2022 ajustada por inflación. Pendiente:
  actualizarla con las cifras de 2025 del reporte de BanRep (el PDF no se pudo leer en el
  entorno de desarrollo).

### Fuentes

- Superfinanciera, *Reporte de Inclusión Financiera 2022*, citado en
  [El Colombiano](https://www.elcolombiano.com/negocios/la-compra-promedio-con-tarjeta-de-credito-en-colombia-es-de-212000-IH21730080).
- Banco de la República,
  [*Reporte de la Infraestructura Financiera e Instrumentos de Pago 2026*](https://www.banrep.gov.co/es/publicaciones-investigaciones/reporte-infraestructura-financiera-instrumentos-pago/2026).
- Banco Central Europeo,
  [*Report on card fraud in 2020 and 2021*](https://www.ecb.europa.eu/press/cardfraud/html/ecb.cardfraudreport202305~5d832d6515.en.html) (2023).
- DANE, proyecciones de población 2023, vía
  [Anexo: Municipios de Colombia por población](https://es.wikipedia.org/wiki/Anexo:Municipios_de_Colombia_por_poblaci%C3%B3n).
- [Asobares sobre la red de Credibanco](https://asobares.org/credibanco-pacta-una-alianza-con-asobares-para-seguir-impulsando-los-pagos-digitales-en-los-comercios-y-micro-comercios-del-pais/).
- Mastercard Colombia,
  [tarifas de intercambio](https://www.mastercard.com/content/dam/mccom/lac/col/colombia-administradora/pdfs/nov25-tarifas-de-intercambio-v2.pdf)
  (referencia pendiente de contrastar; acceso bloqueado).

---

## Feature 2 — Pipeline y datasets analíticos (vigente)

| # | Supuesto | Origen | Dónde |
|---|---|---|---|
| F2-01 | Carga relativa del comercio = MDR efectivo (MDR / volumen) ÷ margen sectorial del comercio | [S §2.4] | `mart_merchant_relative_burden` |
| F2-02 | Los comercios sin compras no aparecen en el mart de carga relativa: sin volumen, la carga no está definida | [P] | `mart_merchant_relative_burden` |
| F2-03 | Aristas del grafo agregadas por mes; cualquier ventana de análisis es la suma de sus meses | [P] | `mart_graph_edges` |
| F2-04 | Las tarifas de la red llegan al warehouse como `network_fees.parquet` (desde la F3, junto con `pricing_terms.parquet`), generadas desde el config: el SQL no tiene tarifas escritas a mano | [P] (diseño) | `params.py`, `stg_network_fees`, `stg_pricing_terms` |
| F2-05 | El P&L en SQL replica la cascada de tarifas de Python (`ips.economics.pnl` desde la F3) y se valida fila a fila contra ella | [P] (diseño) | `fct_transactions`, `tests/test_pipeline.py` |
| F2-06 | Tolerancia de la conservación en SQL: 1e-6 relativo, por aritmética de punto flotante | [P] | `dbt/tests/assert_pnl_conservation.sql` |

---

## Feature 3 — Unit economics y revenue analytics (vigente)

El P&L tiene dos capas. La **cascada de tarifas** reparte el MDR y cumple la identidad de
conservación: emisor bruto + ingreso de la red + adquirente bruto = MDR. Los **costos
operativos** quedan fuera de la identidad y convierten el bruto de cada actor en su
contribución.

### Tarifas de la red y precio al comercio

| # | Supuesto | Origen | Dónde |
|---|---|---|---|
| F3-01 | Fees de la red al adquirente: assessment (scheme fee) de 0,13 % del monto, 80 COP por autorización y 0,45 % en compras cross-border | Assessment [S §1.2: 0,13 %]; autorización [L: APF de Visa, US$0,0195 por autorización de crédito ≈ 80 COP]; cross-border [L: IAF de Visa, 0,45 %] | `economics.yaml: network.acquirer` |
| F3-02 | Fees de la red al emisor: assessment de 0,02 %, 40 COP por autorización y 0,8 % en compras cross-border. Con el del adquirente, los assessments suman 0,15 % del volumen. El recargo cross-border es del orden del ISA de Visa (0,80 %), que en EE. UU. paga el adquirente; aquí lo paga el emisor, que en la práctica lo traslada al tarjetahabiente como comisión por compras en el exterior (fuera del modelo) | Assessments [S §1.6: 0,10–0,15 % del volumen entre ambos lados]; cross-border [L: ISA de Visa, 0,80 %]; reparto entre lados y fee de autorización [P] | `economics.yaml: network.issuer` |
| F3-03 | Modelo de precio del comercio: los adquirentes bancarios (`mixed`) cobran IC++ a sus comercios grandes y medianos; los agregadores, los comercios pequeños y micro y los del exterior pagan blended. Queda en `merchants.pricing_model`, que leen Python y SQL | Modelos [S §1.5]; reglas [P] | `base.yaml: merchants.icpp_tiers`, `population.py` |
| F3-04 | Margen IC++ de 0,20 % + 100 COP: ~0,26 % del monto, en el extremo bajo del rango del adquirente porque lo negocian comercios grandes y medianos | Rango [S §1.6: 0,2–0,7 %]; valor [P] | `economics.yaml: acquirer.icpp_markup` |
| F3-05 | En blended, recargo de 1,5 pp cuando la tarjeta es extranjera para el comercio: los comercios del exterior pagan su tarifa de lista más el recargo en las compras con tarjeta colombiana | [L: Stripe cobra +1,5 % por tarjeta internacional sobre su tarifa de EE. UU.] | `economics.yaml: acquirer.blended_cross_border_surcharge` |
| F3-06 | MDR blended por grupo recalibrado como costo del grupo para el adquirente (interchange y fees de red de su lado, sobre el volumen doméstico en blended) + 0,45 pp, redondeado a 0,05 pp: margen de 0,43–0,47 % por grupo. El fee fijo de autorización sube la tarifa de los grupos de ticket bajo (transporte 1,6 → 2,0 %, alimentos 1,4 → 1,6 %) y la baja en los de ticket alto (viajes 2,9 → 2,7 %, retail 2,3 → 2,2 %) | Rango [S §1.6: 0,2–0,7 %]; tasas [P] | `mcc_groups.yaml: blended_mdr_rate` |

### Costos operativos

| # | Supuesto | Origen | Dónde |
|---|---|---|---|
| F3-07 | Procesamiento del adquirente: 0,10 % + 100 COP por transacción | [P] | `economics.yaml: acquirer.processing` |
| F3-08 | Costos del emisor por producto, en fracción del monto (débito / estándar / premium / comercial): recompensas 0 / 0,4 / 1,2 / 0,8 %; fondeo del período de gracia 0 / 0,20 / 0,25 / 0,30 %; pérdida esperada sobre la compra 0 / 0,30 / 0,15 / 0,10 %; procesamiento 0,10 / 0,15 / 0,20 / 0,20 % | Recompensas [S §1.6: 0,5–1,5 % en premium]; fondeo [S §1.6: 0,1–0,3 %]; procesamiento [S §1.6: 0,1–0,2 %]; pérdida esperada [P] | `economics.yaml: issuer.products` |
| F3-09 | La contribución del emisor es economía de pagos: no incluye ingresos por intereses, pérdidas de la cartera revolvente (que se financian con intereses) ni costos fijos | [S §1.6: el interchange es lo que hace rentable al tarjetahabiente que paga el total cada mes]; alcance [P] | `pnl.py: IssuerPnL` |
| F3-10 | Fraude neto: sin contracargo la pérdida es del emisor; con contracargo pasa al comercio (liability shift). Los contracargos por disputas legítimas también los paga el comercio, cuyo costo es MDR + contracargos | [P], sobre los flags de F1-23 y F1-24 | `pnl.py: IssuerPnL, MerchantCost` |

### Métodos

| # | Supuesto | Origen | Dónde |
|---|---|---|---|
| F3-11 | Revenue bridge: ingreso de la red = volumen × participación de cada región (doméstica o cross-border) × mezcla dentro de la región (producto × grupo × canal) × yield del segmento. El cambio se reparte entre los cuatro factores con valores de Shapley sobre las 16 combinaciones: es exacto y no depende del orden. Un segmento que falta en un período toma el valor del otro | [P], método estándar de atribución | `revenue_bridge.py` |
| F3-12 | Montos en decimales (float64), no en centavos enteros: la conservación se cumple por construcción con tolerancia relativa de 1e-9 | [P] (decisión de diseño) | `pnl.py` |
| F3-13 | Las tablas de referencia (`mccs`, `interchange_table`, `network_fees`, `pricing_terms`) se reescriben desde el config antes de cada `dbt build`: cambiar una tarifa no exige regenerar las transacciones | [P] (diseño) | `params.py`, `tasks.py` |

### Resultado de la calibración (datos sintéticos, semilla 20260910)

- **Yield de la red: 28,1 pb del volumen**: 15,0 de assessment, 7,2 de autorización y 5,9
  cross-border. Las compras cross-border son el 4,7 % del volumen y el 24 % del ingreso de la
  red.
- **Referencia (2025):** Visa ingresó ~29 pb de su volumen de pagos netos de incentivos
  (US$40.000 M sobre US$14 billones) y ~37 pb brutos por servicio, procesamiento y
  transacciones internacionales, antes de US$15.800 M de incentivos. Mastercard ingresó ~31 pb
  netos (US$32.800 M sobre US$10,6 billones de GDV) y ~38 pb brutos por la red de pagos, antes
  de US$20.500 M de rebates e incentivos. Las cifras netas incluyen servicios de valor
  agregado. El modelo no tiene incentivos ni esos servicios, así que su yield se compara con
  la cifra bruta y queda por debajo: sus fees domésticos están dentro de los rangos de la spec
  y el cross-border pesa poco.
- **De cada 100 COP de MDR:** 69,2 van al emisor, 13,4 a la red y 17,4 al adquirente. Tras
  costos, el emisor conserva 33,4 (las recompensas se llevan 18,3) y el adquirente 9,8.
- **Margen bruto del adquirente:** 0,45 % en blended doméstico (0,43–0,47 % por grupo) y
  0,26 % en IC++. Los bancos, con 63–68 % de su volumen en IC++, quedan en 0,31–0,33 %; los
  agregadores, en 0,44–0,45 %.
- **Contribución del emisor:** 0,69 % del volumen en débito y en crédito estándar, 0,52 % en
  premium (las recompensas se comen la mitad de su interchange) y 1,08 % en comercial.

### Consecuencias a tener presentes

- **Micropagos:** en el tramo reducido (F1-26), el fee de autorización del emisor (40 COP)
  supera al interchange (0,20 %). El emisor pierde, antes de costos, en ~22 % de las compras
  con débito y ~14 % de las de crédito estándar. Es un problema de estructura de tarifas que
  el optimizador podrá mover, con un fee de red reducido para montos pequeños o un fijo en el
  interchange.
- **Transporte:** con tickets de ~16.000 COP, el procesamiento fijo (100 COP) deja al
  adquirente con contribución negativa, aunque su margen bruto está en rango.
- **Adquirente extranjero:** solo se ve la parte de su negocio con tarjetas colombianas. Aun
  con el recargo de F3-05, esa parte deja un margen bruto de ~0,10 % y una contribución
  levemente negativa; su negocio con tarjetas de su país está fuera del modelo.
- **IC++ frente a blended:** sobre las mismas compras, IC++ sale 0,16–0,19 pp más barato en
  cada tramo de comercio: la diferencia entre el margen de la tarifa de lista y el negociado.
- **Sin costos de la red:** su ingreso es bruto; no se modelan sus costos operativos.

### Fuentes

- Visa,
  [resultados del año fiscal 2025](https://www.sec.gov/Archives/edgar/data/1403161/000140316125000077/q42025earningsrelease.htm)
  e [informe anual 2025](https://www.sec.gov/Archives/edgar/data/1403161/000130817925000637/v014524-ars.pdf).
- Mastercard,
  [resultados de 2025](https://www.sec.gov/Archives/edgar/data/1141391/000114139126000003/ma12312025-exx991xearnings.htm).
- Tarifas de Visa en EE. UU. (APF, IAF e ISA), resumidas por
  [CardFellow](https://www.cardfellow.com/blog/acquirer-processing-fee) y el
  [Tesoro de Nevada](https://www.nevadatreasurer.gov/uploadedFiles/nevadatreasurergov/content/Merch/Forms/CC_Assessment_Fees.pdf).
- Stripe, [precios públicos](https://stripe.com/pricing): +1,5 % por tarjeta internacional.

---

## Feature 4 — Segmentación con Graph ML (vigente)

La pregunta es de negocio: ¿segmentar comercios por **quién les compra** sirve más, para fijar
precios, que segmentarlos por sus atributos? El baseline es lo que hace la industria (k-means
sobre MCC, tamaño, ticket, canal y mezcla de tarjetas) y el retador es el grafo bipartito
tarjetahabiente-comercio. La respuesta se decide con métricas de negocio, no con la calidad
visual de los clústeres.

### Grafo y representación

| # | Supuesto | Origen | Dónde |
|---|---|---|---|
| F4-01 | El grafo son las compras de los últimos 24 meses; una arista es una tarjeta que le compró a un comercio, con peso igual al número de compras | [P] | `base.yaml: segmentation.window_months`, `build_graph.py` |
| F4-02 | Una tarjeta con más de 200 comercios distintos no señala clientela y se deja fuera; un comercio con menos de 10 compras no entra al clustering, aunque sí recibe vector | [P] | `base.yaml: segmentation`, `features.py` |
| F4-03 | Embeddings espectrales: PPMI sobre los pesos del grafo y SVD truncado a 64 dimensiones, con desplazamiento 1 (el k de muestras negativas de word2vec). Es la factorización matricial que DeepWalk y node2vec aproximan, así que sustituye a `node2vec_embed.py` y `graphsage_embed.py` de la spec | Equivalencia [L: Levy y Goldberg 2014; Qiu et al. 2018]; dimensiones y desplazamiento [P] | `spectral_embed.py` |
| F4-04 | El número de segmentos se elige por silueta entre 4 y 12, sobre una submuestra de 5.000 comercios | [P] | `base.yaml: segmentation.k_range`, `clustering.py` |
| F4-05 | La proyección comercio-comercio solo une comercios que están entre los 10 habituales de una misma tarjeta, con peso mínimo de 3 tarjetas y 30 vecinos por comercio. Sin esa poda la proyección tiene ~10^8 pares y una tarjeta que compra en todas partes conecta todo con todo | [P] | `base.yaml: segmentation`, `build_graph.py` |
| F4-06 | El híbrido concatena atributos y embeddings escalando cada bloque a la misma norma media por fila: sin eso, el bloque con más columnas domina la distancia | [P] | `clustering.py: combine_spaces` |

### Evaluación

| # | Supuesto | Origen | Dónde |
|---|---|---|---|
| F4-07 | La carga relativa es el sustituto del abandono de aceptación hasta la Feature 5, porque la spec la señala como su mejor predictor. La métrica de negocio es el R² fuera de muestra: se reserva el 30 % de los comercios, se predice cada uno con la media de su segmento en el 70 % restante y se compara contra predecir la media global | [S §2.4, §2.5]; diseño [P] | `base.yaml: segmentation.test_fraction`, `evaluate.py` |
| F4-08 | La homogeneidad del interchange efectivo se pondera por volumen, porque ahí está el ingreso; la de la carga relativa va por comercio, porque dejar de aceptar es una decisión por comercio | [P] | `evaluate.py` |
| F4-09 | La clientela latente que plantó el generador (F1-16) se usa solo como diagnóstico del método, nunca como criterio de éxito: el veredicto lo decide la métrica de negocio | [P], criterio de honestidad de CLAUDE.md | `evaluate.py`, notebook 03 |

### Escalera de comparación (Feature 4b)

| # | Supuesto | Origen | Dónde |
|---|---|---|---|
| F4-10 | La referencia de toda la escalera es dejar cada comercio en su propio grupo de MCC, sin modelo ni analítica. Una segmentación que no le gane no se justifica | [P], criterio de honestidad de CLAUDE.md | `baseline_kmeans.py: segment_by_mcc_group` |
| F4-11 | El baseline equilibra sus dos bloques de variables (continuas estandarizadas e indicadores de categoría) a la misma norma media por fila; sin eso, las continuas dominan la distancia y la categoría del comercio, que es la que manda en su economía, casi no cuenta | [P] | `baseline_kmeans.py: feature_matrix` |
| F4-12 | Segmentación supervisada: los segmentos son las hojas de un árbol de regresión sobre la carga relativa, la segmentación por árbol que usa la industria (CART, CHAID). Cada segmento queda descrito por la regla que lo define | [P] | `supervised.py` |
| F4-13 | Todas las cifras fuera de muestra usan la misma partición (`train_test_split`); el árbol se ajusta solo con la mitad de entrenamiento y su número de hojas se elige dentro de esa mitad. La evaluación ordena por `merchant_id` porque la partición es posicional | [P] (diseño) | `evaluate.py`, `supervised.py` |
| F4-14 | El aporte del grafo se mide además con el mismo predictor sobre cada representación, sin el paso de clustering: así ningún método se lleva el crédito de la compresión en vez del de la información | [P] | `evaluate.py: marginal_information` |

### Resultado (datos sintéticos, semilla 20260910)

28.438 comercios de 30.521 con compras, 4,3M aristas y 150.000 tarjetas; la corrida completa
toma ~60 s.

| Método | Segmentos | R² fuera de muestra | Lift en el decil más cargado | NMI con MCC | NMI con clientela |
|---|---|---|---|---|---|
| Grupo de MCC, sin modelo | 12 | 0,740 | 3,63 | 1,000 | 0,033 |
| k-means, atributos | 10 | 0,081 | 1,55 | 0,213 | 0,059 |
| k-means, grafo | 12 | 0,011 | 1,07 | 0,035 | **0,373** |
| k-means, atributos + grafo | 12 | 0,069 | 1,52 | 0,200 | 0,220 |
| Leiden | 8 (55 % de cobertura) | 0,001 | 0,83 | 0,008 | 0,002 |
| **Supervisada, atributos** | 12 | **0,751** | **3,81** | 0,838 | 0,033 |
| Supervisada, atributos + grafo | 12 | 0,751 | 3,81 | 0,838 | 0,033 |

Información marginal con el mismo predictor: atributos 0,751; solo grafo 0,105; atributos +
grafo 0,751, es decir una diferencia de 0,000.

- **El grafo no le gana al baseline, y tampoco aporta sobre los atributos.** Con el mismo
  predictor, sumarle los embeddings deja el R² idéntico. La razón es económica y no del método:
  la carga es MDR efectivo sobre margen sectorial, y el margen es una propiedad del grupo de
  MCC.
- **El techo nunca fue el problema.** Quedarse con el grupo de MCC, sin modelo, explica 0,740 de
  la carga fuera de muestra; el k-means comprimía doce categorías en ocho o diez clústeres y
  perdía justo lo que mueve el objetivo.
- **La segmentación supervisada es la que sirve:** 0,751 con doce hojas, una regla legible por
  segmento y el mejor lift (3,81) para encontrar a los comercios más cargados.
- **El método del grafo funciona sobre estos datos:** solo con el grafo, la segmentación
  recupera la clientela plantada (NMI 0,373 frente a 0,059 del baseline). Esa estructura existe;
  simplemente no es la que decide lo que paga un comercio.
- **Leiden sobre la proyección podada es el más débil:** cubre el 55 % de los comercios y
  explica casi nada. La poda conserva señal de clientela pero pierde cobertura.
- **El veredicto no depende del número de segmentos:** barriendo k de 4 a 12, el grafo queda por
  debajo en todos los casos y la segmentación supervisada por encima en todos.

### Consecuencias a tener presentes

- Los resultados dependen de que el generador haya plantado la estructura que el grafo busca.
  Que la recupere muestra que el método funciona con estos datos, no que los comercios
  colombianos se agrupen así.
- La comparación se decide fuera de muestra a propósito: la homogeneidad dentro del clúster
  siempre mejora al partir en más segmentos, y Leiden elige su propio número de comunidades.
- La Feature 5 traerá la curva de abandono; `evaluate.py` la sumará como una columna más y la
  comparación se rehará con la métrica real. El grafo merece una segunda audiencia ahí: quién
  le compra a quién es exactamente lo que decide a dónde se va el volumen cuando un comercio
  deja de aceptar.
- Los segmentos supervisados sirven para el objetivo con el que se cortaron. Para preguntas de
  sustitución hay que volver a la representación del grafo.

### Fuentes

- Omer Levy y Yoav Goldberg, *Neural Word Embedding as Implicit Matrix Factorization*
  (NeurIPS 2014).
- Jiezhong Qiu et al.,
  [*Network Embedding as Matrix Factorization: Unifying DeepWalk, LINE, PTE, and node2vec*](https://arxiv.org/abs/1710.02971)
  (WSDM 2018).

---

## Feature 5 — Elasticidad y link prediction (vigente)

Dos preguntas que el simulador va a encadenar: **cuánto** volumen se va cuando sube la carga de
un comercio, y **a dónde** se va. Son de naturaleza distinta. La primera no se puede estimar con
estos datos —el generador no plantó ni un solo abandono—, así que la curva es un **supuesto
estructural** con dos anclas propias. La segunda sí se mide: los meses reservados contienen
enlaces reales que ningún método vio.

### Curva de aceptación

| # | Supuesto | Origen | Dónde |
|---|---|---|---|
| F5-01 | La probabilidad anual de que un comercio deje de aceptar la tarjeta es una logística sobre su carga relativa (MDR efectivo ÷ margen sectorial) | Driver [S §2.4, §2.6]; forma funcional [P] | `merchant_acceptance.py` |
| F5-02 | La curva **no se estima**: los datos no tienen eventos de abandono. Sus parámetros —un intercepto por grupo de MCC y una pendiente común— se resuelven por *moment matching* sobre la distribución real de carga de la población, para reproducir dos anclas: 3,5 % de abandono anual y 1,8 pp de aceptación perdida cuando todos los MDR suben 10 % | Método [P]; **las dos cifras son [P]**, sin fuente publicada verificada | `elasticity.yaml: acceptance`, `calibration.py` |
| F5-03 | Cada grupo de MCC tiene su propio nivel de abandono, proporcional a la mediana de su carga relativa (exponente 1) y normalizado para que la población siga en el 3,5 %: un sector que entrega el doble de su margen en MDR abandona el doble. La pendiente es común, y la reacción de cada grupo a la misma carga se escala con la `base_elasticity` de `mcc_groups.yaml`, normalizada por su mediana. Con un solo intercepto para toda la población, la pendiente que exige la segunda ancla volvía la curva un umbral y dejaba a siete de los doce grupos sin respuesta al precio | [P] | `elasticity.yaml: acceptance.level_burden_exponent`, `calibration.py: level_targets`, `params.py: group_sensitivity` |
| F5-04 | Modificadores del logit: +0,35 por unidad de participación no presente, porque ese comercio tiene más alternativas de cobro; −0,60 si se permite el recargo, apagado en el escenario base | Signo del recargo [S §1.8]; signo del no presente y magnitudes [P] | `elasticity.yaml: acceptance.modifiers` |
| F5-05 | Un comercio en IC++ reacciona 1,25 veces más a la misma carga que uno en blended (el `mixed` queda a mitad de camino), porque ve el cambio de interchange directo | Dirección [S §1.5]; magnitud [P] | `merchant_acceptance.py: burden_term` |
| F5-06 | Presión del pago instantáneo: hasta +0,45 en el logit, proporcional a la participación del débito por la fracción en que el ticket medio queda bajo 50.000 COP | Dirección [S §1.9]; umbral [S §2.6]; forma y magnitud [P] | `elasticity.yaml: acceptance.instant_payments` |
| F5-07 | La probabilidad se acota entre 0,05 % y 60 %, y la calibración se hace sobre la curva ya acotada, que es la que consume todo lo demás | [P] | `calibration.py` |

### Respuesta del tarjetahabiente

| # | Supuesto | Origen | Dónde |
|---|---|---|---|
| F5-08 | El gasto con tarjeta responde a la tasa de recompensas como exp(ε · Δtasa), con ε = 0,4 / 0,9 / 1,6 para los segmentos de gasto bajo, medio y alto, y un cambio máximo de ±25 % | [P], sin estimación publicada verificada | `elasticity.yaml: cardholder`, `cardholder_response.py` |

### Link prediction

| # | Supuesto | Origen | Dónde |
|---|---|---|---|
| F5-09 | Partición temporal: los últimos 3 meses de la ventana evalúan y el resto entrena. Un positivo es un par tarjetahabiente-comercio que aparece por primera vez en los meses reservados; las recompras no cuentan | [P] (diseño) | `link_prediction.py: split_by_time` |
| F5-10 | Presupuesto: entrenan 40.000 titulares (el tope se aplica antes de todos los métodos, así que todos ven el mismo grafo), se evalúan 2.000 y los candidatos son los 3.000 comercios más comprados en entrenamiento | [P] (costo en CPU) | `elasticity.yaml: link_prediction` |
| F5-11 | El escalón a vencer es la popularidad dentro de las categorías que el titular ya usa: no usa el grafo, y una representación que no le gane es popularidad disfrazada | [P], criterio de honestidad de CLAUDE.md | `scorers.py: group_popularity_scores` |
| F5-12 | GraphSAGE en PyTorch Geometric: dos capas de 64, atributos en ambos lados (del comercio, su perfil de la F4; del titular, segmento, producto, actividad y ticket medio en entrenamiento), decodificador por producto punto y entropía cruzada con un negativo aleatorio por positivo | Método [L: Hamilton et al. 2017]; arquitectura [P] | `gnn.py` |
| F5-13 | Las aristas de entrenamiento se dividen en paso de mensajes (80 %) y supervisión (20 %), y la supervisión en ajuste (80 %) y validación (20 %). Se conservan los vectores de la época con mejor pérdida de validación; el entrenamiento para cuando esa pérdida lleva 40 épocas sin mejorar o cuando se agota el presupuesto de 400 épocas, lo que llegue primero. Nada de eso mira los meses reservados | [P] (diseño) | `gnn.py: fit_graphsage` |
| F5-14 | El ajuste corre en un hilo con semilla fija: el número de hilos cambia el orden de reducción en punto flotante, y con él el resultado | [P] | `gnn.py: _deterministic` |
| F5-15 | Una ventaja en la escalera solo cuenta si su intervalo bootstrap pareado al 95 % (2.000 remuestreos de los titulares evaluados) no cruza el cero; si lo cruza, el veredicto dice que los dos métodos no se distinguen | [P], criterio de honestidad de CLAUDE.md | `link_prediction.py: ranking_gaps`, `verdict` |

### Redistribución

| # | Supuesto | Origen | Dónde |
|---|---|---|---|
| F5-16 | Los sustitutos de un comercio son sus 10 vecinos más similares en la representación de mayor recall@k entre las dos con vectores de comercio (SVD o GraphSAGE), sin restringirlos a su grupo de MCC | [P] | `link_prediction.py: merchant_substitutes` |
| F5-17 | El 35 % del volumen perdido sale del riel de tarjetas (efectivo, pago instantáneo) y el resto se reparte entre los sustitutos según su similitud; el volumen de un comercio sin sustitutos cuenta entero como fuga | [P] | `elasticity.yaml: redistribution`, `link_prediction.py` |

### Resultado (datos sintéticos, semilla 20260910)

28.438 comercios y 5,8M aristas. Entrenan 40.000 titulares (1,3M aristas) y se evalúan 2.000
sobre 5.047 enlaces nuevos entre 3.000 candidatos. La corrida completa toma ~8 minutos, 7,5 de
ellos en GraphSAGE, y dos corridas separadas dan artifacts idénticos bit a bit.

La curva reproduce las dos anclas exactamente, con una pendiente común β = 12,53 y un intercepto
por grupo:

| Grupo | Mediana de carga relativa | Nivel de abandono | Respuesta a un MDR 10 % más alto, ponderada por volumen |
|---|---|---|---|
| Educación | 0,09 | 1,06 % | +0,06 pp |
| Digital | 0,13 | 1,50 % | +0,18 pp |
| Servicios públicos y telecomunicaciones | 0,16 | 1,93 % | +0,47 pp |
| Salud | 0,18 | 2,15 % | +0,46 pp |
| Retail | 0,18 | 2,16 % | +1,11 pp |
| Entretenimiento | 0,21 | 2,51 % | +1,52 pp |
| Hogar y electrónica | 0,27 | 3,13 % | +1,78 pp |
| Restaurantes | 0,27 | 3,15 % | +2,31 pp |
| Transporte | 0,30 | 3,48 % | +3,09 pp |
| Viajes | 0,48 | 5,69 % | +2,64 pp |
| Combustible | 0,51 | 6,01 % | +4,68 pp |
| Mercado | 0,54 | 6,33 % | +4,84 pp |

Con un solo intercepto para toda la población, la misma calibración daba β = 34,01, dejaba al
87 % de los comercios en el piso de 0,05 % y a ocho grupos con menos de 0,1 pp de respuesta.

| Método | recall@10 | Hit rate@10 | MRR | Cobertura |
|---|---|---|---|---|
| Aleatorio | 0,002 | 0,006 | 0,003 | 99,9 % |
| Popularidad global | 0,038 | 0,090 | 0,030 | 0,6 % |
| Popularidad en las categorías del titular | 0,044 | 0,101 | 0,041 | 4,6 % |
| Embedding PPMI+SVD | 0,049 | 0,104 | 0,036 | 74,5 % |
| **GraphSAGE** | **0,053** | **0,118** | **0,048** | 7,8 % |

| Retador frente a referencia | Diferencia en recall@10 | IC 95 % (bootstrap pareado) | Veredicto |
|---|---|---|---|
| SVD frente a popularidad por categoría | +0,005 | −0,006 a +0,015 | No se distinguen |
| GraphSAGE frente a popularidad por categoría | +0,008 | +0,002 a +0,015 | **Gana** |
| GraphSAGE frente a SVD | +0,004 | −0,007 a +0,015 | No se distinguen |

- **El grafo gana su segunda audiencia, pero por poco y solo con GraphSAGE:** es el único
  escalón que le gana al baseline fuerte con un intervalo que no cruza el cero, y lidera también
  en hit rate y MRR. Frente al SVD de la Feature 4, su ventaja no se distingue del ruido.
- **Sin intervalos, la lectura habría sido otra:** el SVD le gana en promedio a la popularidad
  por categoría, pero su intervalo cruza el cero y pierde en MRR (0,036 contra 0,041): encuentra
  algo más de comercios nuevos, pero los pone más abajo en la lista.
- **Los sustitutos casi no salen de la categoría:** el 99 % de los sustitutos de GraphSAGE
  comparte el grupo de MCC del comercio, contra un 2 % a 27 % si se eligieran al azar.
- **La redistribución cuadra:** de 39.779 millones de COP de volumen esperado perdido en la
  ventana, 25.843 millones pasan a sustitutos y 13.936 millones salen del riel de tarjetas.
- **GraphSAGE terminó por presupuesto y no por paciencia:** su mejor validación fue la de la
  época 392 de 400.

### Consecuencias a tener presentes

- **Los niveles por grupo son una regla, no un dato.** Que un sector con el doble de carga
  abandone el doble es el supuesto que corrige el umbral de la primera versión, no algo que estos
  datos muestren. El exponente vive en `config/elasticity.yaml`: en 0 todos los grupos quedan en el
  mismo nivel, y medida sobre los 6M esa variante también corrige el umbral.
- **Educación casi no responde** (+0,06 pp ante un MDR 10 % más alto), porque su MDR es una
  fracción mínima de su margen. Es coherente con la premisa del modelo, pero un optimizador verá
  poco costo de aceptación en ese grupo.
- **El volumen en riesgo lo lideran viajes y electrónica** (52 %), por su volumen alto con carga
  media o alta, y ya no mercado y combustible, que concentraban el 88 % con un solo intercepto.
- **Los sustitutos refinan dentro de la categoría.** La categoría entra al GNN por los atributos
  del comercio, y para repartir volumen sus vecinos se quedan en el mismo grupo de MCC casi
  siempre.
- **La curva no es evidencia.** Sus dos anclas son cifras propias y nunca vio un comercio dejar
  de aceptar. Antes de usarla fuera del portafolio necesitan fuente publicada; la calibración no
  cambia cuando lleguen, solo se edita `config/elasticity.yaml`.
- **La escalera mide una estructura que plantó el generador.** Gane quien gane, gana
  recuperando clientelas que existen en estos datos, no mostrando cómo compran los
  tarjetahabientes colombianos.
- **Los sustitutos son comercio a comercio.** Los rankings se aprenden de titular a comercio;
  repartir el volumen de un comercio por similitud de vectores aproxima repartir el gasto de cada
  cliente según su propio ranking.
- **El volumen en riesgo es un valor esperado sobre la ventana de 24 meses.** Una probabilidad
  anual por el volumen de la ventana es una escala, no un pronóstico; la Feature 6 lo convierte
  en escenarios con su propio horizonte.

### Fuentes

- William L. Hamilton, Rex Ying y Jure Leskovec, *Inductive Representation Learning on Large
  Graphs* (NeurIPS 2017).
- Matthias Fey y Jan Eric Lenssen, *Fast Graph Representation Learning with PyTorch Geometric*
  (ICLR 2019, workshop on Representation Learning on Graphs and Manifolds).

---

## Feature 0 — Vertical slice (retirada en la Feature 1)

El generador simple de la Feature 0 se retiró. Sus supuestos F0-01 a F0-12 (emisores,
distribuciones, tabla de interchange producto × MCC) y F0-14 (MDR blended por MCC) quedan
reemplazados por F1-01 a F1-29. Siguen vigentes, ahora sobre datos de la Feature 1:

| # | Supuesto | Origen |
|---|---|---|
| F0-04 | Una transacción on-us sigue pagando interchange del rol adquirente al rol emisor, sin neteo por grupo financiero | [P] |
| F0-13 | Scheme fee cobrado al adquirente y pagado desde el MDR (ahora el assessment de F3-01) | [S §1.2] |
| F0-15 | El margen bruto del adquirente es el residuo MDR − interchange − fees de red de su lado; en blended puede ser negativo por transacción y en IC++ es exactamente su margen | [S §1.5] |
| F0-16 | "Ingreso" = flujos brutos que salen del MDR. Desde la Feature 3 la cascada incluye los fees del lado emisor (F3-02) y la contribución resta los costos operativos (F3-07 a F3-10) | Simplificación hasta la Feature 3 |
