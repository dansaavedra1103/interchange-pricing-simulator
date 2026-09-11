-- Hechos de transacción con las tarifas de la red aplicadas fila a fila.
with transactions as (
    select
        *,
        cast(amount_cop as double) as amount,
        cast(date_trunc('month', txn_date) as date) as txn_month,
        -- La tabla de interchange distingue compras domésticas y en comercios del exterior.
        case when cross_border then 'cross_border' else 'domestic' end as region
    from {{ ref('stg_transactions') }}
),

priced as (
    select
        t.*,
        {{ interchange_lookup('t.amount', 'ic') }} as interchange_cop,
        -- Scheme fee: lo que la red cobra al adquirente; es su ingreso real.
        t.amount * fees.scheme_fee_rate + fees.scheme_fee_fixed_cop as scheme_fee_cop,
        -- MDR blended: el comercio paga la tasa de su grupo, sin importar la tarjeta.
        t.amount * m.blended_mdr_rate as mdr_cop
    from transactions as t
    left join {{ ref('stg_interchange_table') }} as ic
        on t.product = ic.product
        and t.mcc_group = ic.mcc_group
        and t.channel = ic.channel
        and t.region = ic.region
    left join {{ ref('stg_mccs') }} as m
        on t.mcc = m.mcc
    cross join {{ ref('stg_network_fees') }} as fees
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
    tokenized,
    cross_border,
    on_us,
    amount_cop,
    fraud_flag,
    chargeback_flag,
    interchange_cop,
    scheme_fee_cop,
    mdr_cop,
    -- El adquirente se queda con el residuo; puede ser negativo en tarjetas premium.
    mdr_cop - interchange_cop - scheme_fee_cop as acquirer_net_cop
from priced
