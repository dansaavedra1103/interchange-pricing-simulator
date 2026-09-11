select
    acquirer_id,
    icpp_markup_rate,
    icpp_markup_fixed_cop,
    blended_cross_border_rate
from {{ source('raw', 'pricing_terms') }}
