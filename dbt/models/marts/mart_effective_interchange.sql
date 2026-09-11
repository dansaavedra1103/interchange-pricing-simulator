-- Interchange efectivo por segmento: cuánto del volumen se va al emisor en cada celda de la
-- tabla, una vez aplicados el fijo y el tramo de micropagos.
select
    txn_month,
    product,
    mcc_group,
    channel,
    region,
    count(*) as n_txns,
    sum(amount_cop) as gdv_cop,
    sum(interchange_cop) as interchange_cop,
    sum(scheme_fee_cop) as scheme_fee_cop,
    sum(mdr_cop) as mdr_cop,
    sum(acquirer_net_cop) as acquirer_net_cop,
    sum(interchange_cop) / sum(amount_cop) as effective_interchange_rate,
    sum(mdr_cop) / sum(amount_cop) as effective_mdr_rate,
    sum(acquirer_net_cop) / sum(amount_cop) as acquirer_margin_rate
from {{ ref('fct_transactions') }}
group by all
