select
    product,
    mcc_group,
    channel,
    region,
    rate,
    fixed_cop,
    small_ticket_max_cop,
    small_ticket_rate,
    small_ticket_fixed_cop,
    valid_from,
    valid_to
from {{ source('raw', 'interchange_table') }}
