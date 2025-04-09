# AdOCTeRA

This repository contains the source code for the paper titled <a href="https://dl.acm.org/doi/abs/10.1145/3652583.3658039">AdOCTeRA: Adaptive Optimization Constraints for improved Text-guided Retrieval of Apartments</a> accepted as the main conference paper at ICMR 2024.

## Similarity Amounts
To have the similarity amount based on the language model used to obtain the metric, it can be downloaded directly from [link](https://drive.google.com/file/d/1iujKbUd76lw58vZ6nI8iaDb-a744MiZ9/view?usp=sharing). Then, the zip file should be extracted to obtain the scenes_relevances, which should be placed in the root directory ('./AdOCTeRA/scenes_relevances') of the project. Also, to regenerate the similarity pickle files, you can use the python files existing in the [scenes_relevances_generation](https://github.com/aliabdari/AdOCTeRA/tree/main/scenes_relevances_generation) directory·

## Train and Evaluation
To train and evaluate the method, the [margin_permutations_checking.py](https://github.com/aliabdari/AdOCTeRA/blob/main/train_evaluation/margin_permutations_checking.py) module should be used.
to simply run the code you can use a command like the below sample command for using one threshold and two margins. 
```
CUBLAS_WORKSPACE_CONFIG=:16:8 python margin_permutations_checking.py -txt_sim --thresh_l 0.25 --thresh_u 0.75 --margin_m 0.30 --margin_h 0.35
```
### Arguments

- -txt_sim: By setting this argument, the train will be done using the similarities obtained using the Language model; otherwise, it will use the structural similarities existing among each pair of apartments.
- --margin_l: It allows to set the lower margin. It accepts a float value and the default value is 0.25.
- --margin_m: It allows to set the medium margin. It accepts a float value and the default value is 0.30.
- --margin_h: It allows to set the high margin. It accepts a float value and the default value is 0.35.
- --thresh_l: It allows to set the lower threshold. It accepts a float value and the default value is 0.25.
- --thresh_u: It allows to set the upper threshold. It accepts a float value and the default value is 0.75.
- --output_feature_size: It specifies the output feature size of the visual model and the text model/ default=256
- --is_bidirectional: Since in some of the modules GRU has been used to process textual descriptions, this argument allows the use of GRU or BiGRU models / default=True
- --num_epochs: Specifies the number of epochs/ default=50
- --batch_size: Specifies the batch size/ default=64
- --lr: Specifies the learning rate/ default=0.008
- --step_size: Since in this work the decay technique of the learning rate has been used this argument specifies after how many epochs the decay occurs / default=27
- --gamma: Accordingly, this argument specifies which percentage the decay in learning rate occurs / default=0.75

## Environment
To run the Experiments we have used Python 3.10, with torch 2.5.0+cu124, with NVIDIA L40S GPU

## Reproducibility
To reproduce the entire results presented in the original paper, you can use four bash files existing in the [train_evaluation](https://github.com/aliabdari/AdOCTeRA/tree/main/train_evaluation). There you can find the following files:

- table1_results_generation.sh
- table2_results_generation.sh
- table3_results_generation.sh
- fig4_generation.sh

To generate each kind of results it is needed to execute the related bash file like:

```
./table1_results_generation.sh
```
The results will go automatically in a new directory called generate_results
Note: If the bash file is not executable in your system, you can run the following command
```
chmod +x table1_results_generation.sh
```
