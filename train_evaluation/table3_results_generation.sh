#!/bin/bash

#result=$(CUBLAS_WORKSPACE_CONFIG=:16:8 python3.10 tests.py -simmodel distilroberta --margin_m 0.30 --margin_h 0.35 --thresh_l 0.25 --thresh_u 0.75)
#echo "Result from Python: $result"


thresh_l_values=(0.25 0.35 0.40 0.55 0.55 0.35 0.35 0.35)
thresh_u_values=(0.75 0.75 0.75 0.75 0.75 0.50 0.75 0.90)

output_file="results_table_3.csv"
echo "t_l,t_u,R1,R5,R10,MedR" > $output_file


echo "Generating Results of the Table 1 of the Paper ... "
echo
echo "Running with the S2 similarity method ..."
result=$(CUBLAS_WORKSPACE_CONFIG=:16:8 python3.10 tests.py -custom_margin)
echo "Results of Three Run: $result"
r1=$(echo $result | cut -d',' -f1)
r5=$(echo $result | cut -d',' -f2)
r10=$(echo $result | cut -d',' -f3)
medr=$(echo $result | cut -d',' -f7)

echo "0.25,0.25,$r1,$r5,$r10,$medr" >> $output_file

echo "All tests completed!"