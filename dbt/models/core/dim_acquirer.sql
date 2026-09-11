select
    acquirer_id,
    acquirer_type,
    country,
    bank_group,
    pricing_model
from {{ ref('stg_acquirers') }}
