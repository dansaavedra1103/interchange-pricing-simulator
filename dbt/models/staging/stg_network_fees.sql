select
    issuer_assessment_rate,
    issuer_authorization_fee_cop,
    issuer_cross_border_rate,
    acquirer_assessment_rate,
    acquirer_authorization_fee_cop,
    acquirer_cross_border_rate
from {{ source('raw', 'network_fees') }}
