#!/bin/bash

threshold_l_values=(0.25 0.35 0.45 0.35 0.35)
threshold_u_values=(0.75 0.75 0.75 0.50 0.90)

margin_m_values=(0.28 0.30 0.30 0.35 0.35 0.40 0.40)
margin_h_values=(0.31 0.35 0.40 0.45 0.50 0.55 0.60)

DIR="generated_results"

if [ ! -d "$DIR" ]; then
    mkdir -p "$DIR"
    echo "Directory $DIR created successfully."
fi

output_file="generated_results/results_table_2_tmp.csv"
echo "m1,m2,m3,t_l,t_u,MedR" > $output_file

echo "Generating Results of the Table 2 of the Paper ... "
echo

for ((m=0; m<${#margin_m_values[@]}; m++)); do
    margin_m=${margin_m_values[$m]}
    margin_h=${margin_h_values[$m]}

    echo "Using margin_l=0.25 margin_m=$margin_m and margin_h=$margin_h"

    for ((t=0; t<${#threshold_l_values[@]}; t++)); do
        thresh_l=${threshold_l_values[$t]}
        thresh_u=${threshold_u_values[$t]}

        echo "Running test with --margin_l=0.25 --margin_m=$margin_m, --margin_h=$margin_h, --thresh_l=$thresh_l, and --thresh_u=$thresh_u ..."

        result=$(CUBLAS_WORKSPACE_CONFIG=:16:8 python repro_run.py \
            -txt_sim \
            --margin_m $margin_m \
            --margin_h $margin_h \
            --thresh_l $thresh_l \
            --thresh_u $thresh_u)

        echo "Results of Three Run:"
        echo "ds_r1, ds_r5, ds_r10, sd_r1, sd_r5, sd_r10, ds_medr, sd_medr"
        echo "$result"
        r1=$(echo $result | cut -d',' -f1)
        r5=$(echo $result | cut -d',' -f2)
        r10=$(echo $result | cut -d',' -f3)
        medr=$(echo $result | cut -d',' -f7)

        echo "0.25,$margin_m,$margin_h,$thresh_l,$thresh_u,$medr" >> $output_file
        echo "--------------------------------------"
    done
done
python repro_csv.py
echo "All tests completed!"
