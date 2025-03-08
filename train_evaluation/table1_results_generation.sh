#!/bin/bash

thresh_l_values=(0.25 0.35 0.45 0.55 0.35 0.35 0.35)
thresh_u_values=(0.75 0.75 0.75 0.75 0.50 0.75 0.90)

output_file="results_table_1_no_seed.csv"
echo "t_l,t_u,R1,R5,R10,MedR" > $output_file


echo "Generating Results of the Table 1 of the Paper ... "
echo
echo "Running test of the baseline ..."
result=$(python3.10 repro_baseline.py)
echo "Results of Three Run: $result"
r1=$(echo $result | cut -d',' -f1)
r5=$(echo $result | cut -d',' -f2)
r10=$(echo $result | cut -d',' -f3)
medr=$(echo $result | cut -d',' -f7)
echo "0.25,0.25,$r1,$r5,$r10,$medr" >> $output_file
echo
for i in {0..6}; do
    thresh_l=${thresh_l_values[$i]}
    thresh_u=${thresh_u_values[$i]}

    echo "Running test with --thresh_l=$thresh_l and --thresh_u=$thresh_u ..."

    result=$(python3.10 repro_run_senza_seed.py \
        -custom_margin \
        -txt_sim \
        --margin_m 0.30 \
        --margin_h 0.35 \
        --thresh_l $thresh_l \
        --thresh_u $thresh_u)
    echo "Results of Three Run: $result"
    r1=$(echo $result | cut -d',' -f1)
    r5=$(echo $result | cut -d',' -f2)
    r10=$(echo $result | cut -d',' -f3)
    medr=$(echo $result | cut -d',' -f7)

    echo "$thresh_l,$thresh_u,$r1,$r5,$r10,$medr" >> $output_file
    echo "Completed run $((i+1))/8"
    echo "--------------------------------------"
done

echo "All tests completed!"