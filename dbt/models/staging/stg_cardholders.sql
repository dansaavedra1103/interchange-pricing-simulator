select
    cardholder_id,
    issuer_id,
    product,
    country,
    city,
    spend_segment,
    cross_border_propensity
from {{ source('raw', 'cardholders') }}
