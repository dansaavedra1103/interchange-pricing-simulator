-- Interchange efectivo y rendimiento de la red por segmento: cuánto del volumen llega al
-- emisor y a la red en cada celda de la tabla, una vez aplicados fijos y tramos.
select
    txn_month,
    product,
    mcc_group,
    channel,
    region,
    count(*) as n_txns,
    sum(amount_cop) as gdv_cop,
    sum(interchange_cop) as interchange_cop,
    sum(network_revenue_cop) as network_revenue_cop,
    sum(mdr_cop) as mdr_cop,
    sum(issuer_gross_cop) as issuer_gross_cop,
    sum(acquirer_gross_cop) as acquirer_gross_cop,
    sum(interchange_cop) / sum(amount_cop) as effective_interchange_rate,
    sum(network_revenue_cop) / sum(amount_cop) as net_revenue_yield,
    sum(mdr_cop) / sum(amount_cop) as effective_mdr_rate,
    sum(acquirer_gross_cop) / sum(amount_cop) as acquirer_margin_rate
from {{ ref('fct_transactions') }}
group by all
