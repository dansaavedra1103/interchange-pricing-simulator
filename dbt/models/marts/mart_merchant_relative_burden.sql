-- Carga relativa: fracción del margen del comercio que se va en MDR (spec §2.4). Es el
-- principal predictor de abandono de la aceptación en el simulador.
with volume as (
    select
        merchant_id,
        count(*) as n_txns,
        sum(amount_cop) as gdv_cop,
        sum(mdr_cop) as mdr_cop
    from {{ ref('fct_transactions') }}
    group by merchant_id
)

select
    m.merchant_id,
    m.mcc,
    m.mcc_group,
    m.country,
    m.city,
    m.size_tier,
    m.acquirer_id,
    m.pricing_model,
    v.n_txns,
    v.gdv_cop,
    v.mdr_cop,
    v.mdr_cop / v.gdv_cop as effective_mdr_rate,
    m.sector_margin,
    v.mdr_cop / v.gdv_cop / m.sector_margin as relative_burden
from volume as v
inner join {{ ref('dim_merchant') }} as m
    on v.merchant_id = m.merchant_id
