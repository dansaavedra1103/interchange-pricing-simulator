select
    merchant_id,
    mcc,
    mcc_group,
    country,
    city,
    channel,
    size_tier,
    size_weight,
    acquirer_id,
    pricing_model,
    sector_margin
from {{ source('raw', 'merchants') }}
