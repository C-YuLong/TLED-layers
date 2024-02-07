# set TOP {ubit8_designware appro_42 appro_fulladder appro_halfadder Appro_multi Dadda_PPAM_1_3 Dadda_PPAM_2_2 multiplier_2x2 multiplier_4x4 multiplier_8x8 optimal1 optimal3 RoBA SDLC SiEi6 SiEi7 TOSAM_h3_t3 TOSAM_h3_t4 TOSAM_h3_t5 TOSAM_h3_t6 TOSAM_h3_t7 TOSAM_h4_t3 TOSAM_h4_t4 TOSAM_h4_t5 TOSAM_h4_t6 TOSAM_h4_t7 TOSAM_h5_t3 TOSAM_h5_t4 TOSAM_h5_t5 TOSAM_h5_t6 TOSAM_h5_t7 wallacetreev
# }

# set TOP {column_approx_5 column_approx_6}
# set TOP {column_approx_2_8 column_approx_3_8 column_approx_4_8 column_approx_5_8 column_approx_6_8 column_approx_7_8}
# set TOP {column_approx_8_8 column_approx_9_8 column_approx_10_8 column_approx_11_8}
set TOP {column_approx_1_8}
# set TOP {lenet5}

foreach name $TOP {

set SRC_FILE                scripts/read.tcl
set CONSTRAINT_FILE         scripts/constraints_comb.tcl

set CONSTRAINT_VIOLATION    reports/other/${name}_vio.rpt
set TIMING_RPT              reports/other/${name}_timing.rpt
set AREA_RPT                reports/other/${name}_area.rpt
set POWER_RPT		        reports/other/${name}_power.rpt
set QoR_RPT                 reports/other/${name}_qor.rpt
set SAIF_RPT                reports/other/${name}_saif.rpt

set NETLIST                 outputs/other/${name}_gate_v.v
set DDC                     outputs/other/${name}_ddc.ddc
set SDF                     outputs/other/${name}_sdf.sdf


source $SRC_FILE

link

uniquify

source $CONSTRAINT_FILE

set_wire_load_mode top 
compile_ultra
# compile

report_constraints  > $CONSTRAINT_VIOLATION
report_timing       > $TIMING_RPT
report_area         > $AREA_RPT
report_power        > $POWER_RPT
report_qor          > $QoR_RPT
report_saif         > $SAIF_RPT

write_file -hierarchy -format verilog -output $NETLIST
write_file -hierarchy -format ddc -output $DDC
write_sdf $SDF
}

quit