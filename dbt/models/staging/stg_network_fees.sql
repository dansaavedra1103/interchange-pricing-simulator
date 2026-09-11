select
    scheme_fee_rate,
    scheme_fee_fixed_cop
from {{ source('raw', 'network_fees') }}
