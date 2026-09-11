-- Hechos de transacción con la cascada de tarifas de la red aplicada fila a fila. Replica
-- ips.economics.pnl (mismas sumas en el mismo orden) y un pytest lo verifica fila a fila.
with transactions as (
    select
        *,
        cast(amount_cop as double) as amount,
        cast(date_trunc('month', txn_date) as date) as txn_month,
        -- La tabla de interchange distingue compras domésticas y en comercios del exterior.
        case when cross_border then 'cross_border' else 'domestic' end as region
    from {{ ref('stg_transactions') }}
),

fees as (
    select
        t.*,
        m.pricing_model,
        mc.blended_mdr_rate,
        pt.icpp_markup_rate,
        pt.icpp_markup_fixed_cop,
        pt.blended_cross_border_rate,
        {{ interchange_lookup('t.amount', 'ic') }} as interchange_cop,
        -- Fees de la red a cada lado: assessment sobre el monto, fee fijo por autorización y
        -- recargo cross-border cuando el comercio está en otro país.
        t.amount * nf.issuer_assessment_rate as issuer_assessment_cop,
        nf.issuer_authorization_fee_cop as issuer_authorization_cop,
        case when t.cross_border then t.amount * nf.issuer_cross_border_rate else 0.0 end
            as issuer_cross_border_cop,
        t.amount * nf.acquirer_assessment_rate as acquirer_assessment_cop,
        nf.acquirer_authorization_fee_cop as acquirer_authorization_cop,
        case when t.cross_border then t.amount * nf.acquirer_cross_border_rate else 0.0 end
            as acquirer_cross_border_cop
    from transactions as t
    left join {{ ref('stg_interchange_table') }} as ic
        on t.product = ic.product
        and t.mcc_group = ic.mcc_group
        and t.channel = ic.channel
        and t.region = ic.region
    left join {{ ref('stg_mccs') }} as mc
        on t.mcc = mc.mcc
    left join {{ ref('stg_merchants') }} as m
        on t.merchant_id = m.merchant_id
    left join {{ ref('stg_pricing_terms') }} as pt
        on t.acquirer_id = pt.acquirer_id
    cross join {{ ref('stg_network_fees') }} as nf
),

sides as (
    select
        *,
        issuer_assessment_cop + issuer_authorization_cop + issuer_cross_border_cop
            as issuer_network_fee_cop,
        acquirer_assessment_cop + acquirer_authorization_cop + acquirer_cross_border_cop
            as acquirer_network_fee_cop
    from fees
),

priced as (
    select
        *,
        -- MDR: en blended el comercio paga la tasa plana de su grupo, más un recargo cuando la
        -- tarjeta es extranjera para él; en IC++ paga el interchange y los fees de red del
        -- adquirente tal cual, más el margen del adquirente.
        case
            when pricing_model = 'icpp'
                then interchange_cop + acquirer_network_fee_cop
                + (amount * icpp_markup_rate + icpp_markup_fixed_cop)
            else amount * (
                blended_mdr_rate
                + case when cross_border then blended_cross_border_rate else 0.0 end
            )
        end as mdr_cop
    from sides
)

select
    txn_id,
    txn_date,
    txn_month,
    cardholder_id,
    merchant_id,
    issuer_id,
    acquirer_id,
    product,
    mcc,
    mcc_group,
    channel,
    region,
    pricing_model,
    tokenized,
    cross_border,
    on_us,
    amount_cop,
    fraud_flag,
    chargeback_flag,
    interchange_cop,
    issuer_network_fee_cop,
    acquirer_network_fee_cop,
    issuer_network_fee_cop + acquirer_network_fee_cop as network_revenue_cop,
    issuer_assessment_cop + acquirer_assessment_cop as network_assessment_cop,
    issuer_authorization_cop + acquirer_authorization_cop as network_authorization_cop,
    issuer_cross_border_cop + acquirer_cross_border_cop as network_cross_border_cop,
    mdr_cop,
    -- El emisor se queda el interchange neto de sus fees de red.
    interchange_cop - issuer_network_fee_cop as issuer_gross_cop,
    -- El adquirente se queda el residuo; en IC++ es exactamente su margen.
    mdr_cop - interchange_cop - acquirer_network_fee_cop as acquirer_gross_cop
from priced
