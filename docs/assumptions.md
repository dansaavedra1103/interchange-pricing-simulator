# Supuestos del generador y del modelo económico

Los datos del proyecto son **sintéticos**: no existen datos públicos reales de interchange.
Cada supuesto se registra aquí en el momento de introducirlo, con su origen:

- **[spec/literatura]**: tomado de `docs/spec.md` (que resume práctica de la industria y
  regulación) o de literatura citada.
- **[supuesto propio]**: elección de diseño sin calibración externa. Debe leerse como
  ilustrativa, nunca como hallazgo empírico.

Los valores exactos viven en `config/base.yaml`; este documento explica el porqué.

---

## Feature 0 — Vertical slice

### Estructura de la red

| # | Supuesto | Origen | Dónde |
|---|---|---|---|
| F0-01 | 6 emisores ficticios: 4 bancos y 2 fintech, con participación de mercado 32 / 24 / 16 / 10 / 12 / 6 % | [supuesto propio] | `issuers` |
| F0-02 | 4 adquirentes ficticios: 2 bancos que también emiten (grupos A y B) y 2 agregadores, con 30 / 25 / 25 / 20 % de los comercios | [supuesto propio] | `acquirers` |
| F0-03 | El adquirente se asigna **al comercio**, solo con las participaciones de los adquirentes e independiente del emisor del tarjetahabiente. `on_us` = emisor y adquirente del mismo `bank_group` | [spec §1.1] | `simple_generator.generate_merchants` |
| F0-04 | Una transacción on-us sigue pagando interchange del rol adquirente al rol emisor: no se netea a nivel de grupo financiero | [supuesto propio] | `simple_pnl.compute_pnl` |

### Distribuciones

| # | Supuesto | Origen | Dónde |
|---|---|---|---|
| F0-05 | MCC del comercio categórico entre 10 MCC; supermercados (20 %) y restaurantes (16 %) son los más frecuentes | Dirección de [spec §2.3] (retail y alimentos dominan); valores [supuesto propio] | `mccs[].merchant_share` |
| F0-06 | Tamaño del comercio: peso de volumen ~ LogNormal(0, σ = 1,0). Cada transacción elige comercio con probabilidad proporcional a ese peso | Forma log-normal de [spec §2.3]; σ [supuesto propio] | `distributions.merchant_size_sigma` |
| F0-07 | Emisor del tarjetahabiente ~ Categórica(participación de mercado). Producto ~ Categórica(mix del tipo de emisor): bancos 55 / 30 / 15 % (débito / crédito estándar / premium), fintech 80 / 17 / 3 % | Dirección de [spec §2.3] (bancos más premium, fintech más débito); valores [supuesto propio] | `product_mix` |
| F0-08 | Actividad del tarjetahabiente ~ LogNormal(0, σ = 0,7), independiente del producto | [supuesto propio] | `distributions.cardholder_activity_sigma` |
| F0-09 | Monto ~ LogNormal con mediana por MCC (25.000 a 600.000 COP) y σ en log de 0,6 a 1,0, redondeado a pesos, con piso de 1.000 COP. El producto no afecta el monto | [supuesto propio]. Desviación deliberada de "uniformes/normales" (spec §3.3): una normal sobre montos genera valores negativos y la asimetría a la derecha es la forma habitual del ticket | `mccs[].median_ticket_cop`, `ticket_sigma`, `distributions.min_amount_cop` |
| F0-10 | Fecha uniforme en 2025, sin estacionalidad semanal ni mensual | Simplificación F0 (la Feature 1 agrega estacionalidad) | `dates` |
| F0-11 | Tarjetahabiente y comercio se sortean de forma independiente en cada transacción: no hay afinidad, así que el grafo no tiene comunidades | Simplificación F0 (la Feature 1 agrega el proceso de afinidad de spec §2.3) | `simple_generator.generate_transactions` |

### Tarifas y P&L

| # | Supuesto | Origen | Dónde |
|---|---|---|---|
| F0-12 | Interchange = tabla única producto × MCC (3 × 10), porcentual + fijo. Débito 0,3–1,1 % (0,3–0,4 % en supermercados, gasolina y servicios públicos), crédito estándar 1,0–2,0 %, premium 1,3–2,5 %. Fijo = 0 en toda la grilla | Rangos de [spec §2.3]; valor de cada celda [supuesto propio] | `pricing.interchange_table` |
| F0-13 | Scheme fee de 0,13 % del monto, cobrado al adquirente y pagado desde el MDR | [spec §1.2] (ejemplo) dentro del rango 0,10–0,15 % de [spec §1.6] | `pricing.scheme_fee` |
| F0-14 | Todos los comercios pagan MDR **blended** por MCC (1,3–2,3 %), calibrado para que el margen promedio del adquirente quede en 0,2–0,7 % del volumen | Rango de margen de [spec §1.6]; tasas [supuesto propio] | `pricing.blended_mdr_rate` |
| F0-15 | El adquirente se queda con el residuo MDR − interchange − scheme fee. En blended puede ser negativo por transacción (tarjetas premium) | [spec §1.5] (en blended el adquirente absorbe la variabilidad del mix) | `simple_pnl.compute_pnl` |
| F0-16 | "Ingreso" = flujos brutos que salen del MDR. Sin costos del emisor (recompensas, fraude, fondeo, pérdida de crédito), sin assessments del lado emisor, sin tarifas por autorización ni cross-border | Simplificación F0 (Feature 3) | `simple_pnl` |
| F0-17 | Sin canal (CP/CNP), sin región ni cross-border, sin fraude ni contracargos, sin IC++ | Simplificación F0 (Features 1 y 3) | — |

### Consecuencia a tener presente

Con un scheme fee puramente porcentual y sin respuesta del volumen al precio, el net revenue
yield de la red es **13 pb por construcción**. El número que produce la Feature 0 valida la
tubería de punta a punta; no dice nada sobre cómo fijar el interchange.
