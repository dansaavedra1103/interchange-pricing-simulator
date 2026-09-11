select
    issuer_id,
    issuer_type,
    size,
    country,
    bank_group,
    market_share
from {{ source('raw', 'issuers') }}
