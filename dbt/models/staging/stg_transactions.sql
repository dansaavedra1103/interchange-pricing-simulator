select
    txn_id,
    txn_date,
    cardholder_id,
    merchant_id,
    issuer_id,
    acquirer_id,
    product,
    mcc,
    mcc_group,
    channel,
    tokenized,
    cross_border,
    on_us,
    amount_cop,
    fraud_flag,
    chargeback_flag
from {{ source('raw', 'transactions') }}
