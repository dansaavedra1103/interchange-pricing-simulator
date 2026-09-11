{#-
    Interchange fee of a transaction joined to its cell of the interchange table:
    `amount * rate + fixed`, or the reduced small-ticket tier when the cell has one and the
    amount does not exceed its cap. Mirrors ips.data_gen.interchange_table.apply_interchange.

    Tramo de micropagos: en montos pequeños el fijo dispararía el porcentaje efectivo, por eso
    la red fija una tarifa reducida (spec §1.4).
-#}
{% macro interchange_lookup(amount, cell) -%}
    case
        when {{ cell }}.small_ticket_max_cop is not null
            and {{ amount }} <= {{ cell }}.small_ticket_max_cop
            then {{ amount }} * {{ cell }}.small_ticket_rate + {{ cell }}.small_ticket_fixed_cop
        else {{ amount }} * {{ cell }}.rate + {{ cell }}.fixed_cop
    end
{%- endmacro %}
