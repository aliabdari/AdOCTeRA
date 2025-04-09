#!/bin/bash


DIR="generated_results"

if [ ! -d "$DIR" ]; then
    mkdir -p "$DIR"
    echo "Directory $DIR created successfully."
fi

output_file="generated_results/results_table_3.csv"
echo "Method,R1,R5,R10,MedR" > $output_file


echo "Running test of the baseline ..."
result=$(CUBLAS_WORKSPACE_CONFIG=:16:8 python repro_baseline.py)
echo "Results of Three Run:"
echo "ds_r1, ds_r5, ds_r10, sd_r1, sd_r5, sd_r10, ds_medr, sd_medr"
echo "$result"
r1=$(echo $result | cut -d',' -f1)
r5=$(echo $result | cut -d',' -f2)
r10=$(echo $result | cut -d',' -f3)
medr=$(echo $result | cut -d',' -f7)
echo "baseline,$r1,$r5,$r10,$medr" >> $output_file

echo "Generating Results of the Table 3 of the Paper ... "
echo
echo "Running test S2 using Structural Similarity..."
result=$(CUBLAS_WORKSPACE_CONFIG=:16:8 python repro_run.py)
echo "Results of Three Run:"
echo "ds_r1, ds_r5, ds_r10, sd_r1, sd_r5, sd_r10, ds_medr, sd_medr"
echo "$result"
r1=$(echo $result | cut -d',' -f1)
r5=$(echo $result | cut -d',' -f2)
r10=$(echo $result | cut -d',' -f3)
medr=$(echo $result | cut -d',' -f7)

echo "S2,$r1,$r5,$r10,$medr" >> $output_file


echo "Running test S(ours)  --margin_m 0.30 --margin_h 0.35 --thresh_l 0.35 --thresh_u 0.75..."
result=$(CUBLAS_WORKSPACE_CONFIG=:16:8 python repro_run.py \
    -txt_sim \
    --margin_m 0.28 \
    --margin_h 0.31 \
    --thresh_l 0.25 \
    --thresh_u 0.75)

echo "Results of Three Run:"
echo "ds_r1, ds_r5, ds_r10, sd_r1, sd_r5, sd_r10, ds_medr, sd_medr"
echo "$result"

r1=$(echo $result | cut -d',' -f1)
r5=$(echo $result | cut -d',' -f2)
r10=$(echo $result | cut -d',' -f3)
medr=$(echo $result | cut -d',' -f7)

echo "S(ours),$r1,$r5,$r10,$medr" >> $output_file

echo "All tests completed!"
