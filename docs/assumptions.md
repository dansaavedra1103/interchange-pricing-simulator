# Supuestos del generador y del modelo económico

Los datos del proyecto son **sintéticos**: no existen datos públicos reales de interchange.
Cada supuesto se registra aquí en el momento de introducirlo, con su origen:

- **[D] dato público**: cifra oficial usada directamente (p. ej., población DANE).
- **[L] literatura**: tomado de un informe o publicación citada abajo.
- **[S] spec**: tomado de `docs/spec.md`, que resume práctica de la industria y regulación.
- **[P] supuesto propio**: elección de diseño sin calibración externa. Debe leerse como
  ilustrativa, nunca como hallazgo empírico.

Los valores exactos viven en `config/base.yaml`, `config/mcc_groups.yaml` y
`config/interchange_table.yaml`; este documento explica el porqué.

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

### P&L transitorio (`simple_pnl`)

| # | Supuesto | Origen | Dónde |
|---|---|---|---|
| F1-28 | Scheme fee de 0,13 % del monto, cobrado al adquirente desde el MDR | [S §1.2]; rango 0,10–0,15 % [S §1.6] | `base.yaml: network` |
| F1-29 | Todo comercio paga el MDR blended de su grupo, calibrado como costo del grupo + ~0,45 pp: margen del adquirente de 0,44–0,51 % por grupo | Rango de margen [S §1.6: 0,2–0,7 %]; tasas [P] | `mcc_groups.yaml: blended_mdr_rate` |
| F1-30 | En compras cross-border el comercio extranjero también paga la tarifa blended de su grupo, y el adquirente extranjero puede perder. IC++, tarifas cross-border y de autorización y costos del emisor llegan con la Feature 3 | [P] | `simple_pnl.py` |

### Consecuencias a tener presentes

- Con un scheme fee puramente porcentual, el net revenue yield de la red es **13 pb por
  construcción** hasta la Feature 3.
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

## Feature 0 — Vertical slice (retirada en la Feature 1)

El generador simple de la Feature 0 se retiró. Sus supuestos F0-01 a F0-12 (emisores,
distribuciones, tabla de interchange producto × MCC) y F0-14 (MDR blended por MCC) quedan
reemplazados por F1-01 a F1-29. Siguen vigentes, ahora sobre datos de la Feature 1:

| # | Supuesto | Origen |
|---|---|---|
| F0-04 | Una transacción on-us sigue pagando interchange del rol adquirente al rol emisor, sin neteo por grupo financiero | [P] |
| F0-13 | Scheme fee cobrado al adquirente y pagado desde el MDR (ahora F1-28) | [S §1.2] |
| F0-15 | El margen del adquirente es el residuo MDR − interchange − scheme fee; puede ser negativo por transacción | [S §1.5] |
| F0-16 | "Ingreso" = flujos brutos que salen del MDR; sin costos del emisor ni assessments del lado emisor | Simplificación hasta la Feature 3 |
