-- Aristas del grafo tarjetahabiente-comercio por mes; cualquier ventana es la suma de meses.
select
    txn_month,
    cardholder_id,
    merchant_id,
    count(*) as n_txns,
    sum(amount_cop) as amount_cop
from {{ ref('fct_transactions') }}
group by all
