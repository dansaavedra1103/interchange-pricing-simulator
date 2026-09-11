select
    m.merchant_id,
    m.mcc,
    mc.mcc_name,
    m.mcc_group,
    m.country,
    m.city,
    m.channel,
    m.size_tier,
    m.size_weight,
    m.acquirer_id,
    m.sector_margin
from {{ ref('stg_merchants') }} as m
left join {{ ref('stg_mccs') }} as mc
    on m.mcc = mc.mcc
