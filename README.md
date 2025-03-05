# AdOCTeRA

This repository contains the source code for the paper titled <a href="https://dl.acm.org/doi/abs/10.1145/3652583.3658039">AdOCTeRA: Adaptive Optimization Constraints for improved Text-guided Retrieval of Apartments</a> accepted as the main conference paper at ICMR 2024.

## Similarity Amounts
To have the similarity amount based on the language model used to obtain the metric it can be downloaded directly from [link](https://drive.google.com/file/d/14c4vAY045WWp2u7sAJWMwIti2TQupRON/view?usp=sharing). Then the zip file should be extracted to obtain the scenes_relevances, which should be placed in the root directory ('./AdOCTeRA/scenes_relevances') of the project.

## Train and Evaluation
To train and evaluate the method the [margin_permutations_checking.py](https://github.com/aliabdari/AdOCTeRA/blob/main/train_evaluation/margin_permutations_checking.py) module should be used.
to simply run the code you can use a command like the below sample command for using one thresholds and two margins. 
```
python margin_permutations_checking.py -custom_margin --thresh_l 0.25 --thresh_u 0.75 --margin_m 0.30 --margin_h 0.35
```
### Arguments

- -custom_margin: by using this argument it will use custom margins and thresholds and without putting this argument the code will be run in the default mode which does not have any threshold and uses one margin
- --margin_l: It allows to set the lower margin. It accepts a float value and the default value is 0.25.
- --margin_l: It allows to set the medium margin. It accepts a float value and the default value is 0.30.
- --margin_m: It allows to set the high margin. It accepts a float value and the default value is 0.35.
- --thresh_h: It allows to set the lower threshold. It accepts a float value and the default value is 0.25.
- --thresh_u: It allows to set the upper threshold. It accepts a float value and the default value is 0.75. 
- -simmodel: with this switch, we can specify the similarity model that will be used to measure the similarity between descriptions it could be 'distilroberta' or 'MiniLM' or 'gte-large'

