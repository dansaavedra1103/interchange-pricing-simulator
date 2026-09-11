select
    c.cardholder_id,
    c.issuer_id,
    i.issuer_type,
    c.product,
    c.country,
    c.city,
    c.spend_segment,
    c.cross_border_propensity
from {{ ref('stg_cardholders') }} as c
left join {{ ref('stg_issuers') }} as i
    on c.issuer_id = i.issuer_id
