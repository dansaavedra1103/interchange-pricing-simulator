select
    mcc,
    name as mcc_name,
    mcc_group,
    sector_margin,
    base_elasticity,
    blended_mdr_rate
from {{ source('raw', 'mccs') }}
