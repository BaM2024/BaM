# Towards True Multi-interest Recommendation: Enhanced Training Schemes for Balanced Interest Learning

This project is a pytorch implementation of 'Towards True Multi-interest Recommendation: Enhanced Training Schemes for Balanced Interest Learning'.
BaM (<U/>Ba</U>lanced Interest Learning for <U/>M</U>ulti-interest Recommendation) is an effective and generally applicable training scheme for balanced learning of multi-interest and it achieves up to 21.23% higher accuracy in sequential recommendation compared to the best competitor, resulting in the state-of-the-art performance.
This project provides executable source code with adjustable arguments and preprocessed datasets used in the paper.

## Prerequisites

- Python 3.8+
- [PyTorch](https://pytorch.org/) 1.12.1
- [NumPy](https://numpy.org/)
- [tqdm](https://tqdm.github.io/)

## Usage

There are 3 folders and each consists of:
- data: preprocessed datasets
- runs: pre-trained models for each dataset
- src: source codes

You can run a demo script 'demo.sh' to compare the performance of BaM_p (BaM with correlation score using simple inner-product (Equation (8) in the paper) and BaM_l (BaM with correlation score using additional linear layers (Equation (9) in the paper) in Movies & TV dataset by evaluating pre-trained model.
The result looks as follows:
```
users: 22747
items: 17848
interactions: 841607
model loaded from ./runs/movies/bam_p_1/
test: 22747it [00:15, 1474.88it/s]
test recall[@10, @20]: [0.0819, 0.1153], test nDCG[@10, @20]: [0.0475, 0.0559]

users: 22747
items: 17848
interactions: 841607
model loaded from ./runs/movies/bam_l_1/
test: 22747it [00:15, 1467.02it/s]
test recall[@10, @20]: [0.0801, 0.1141], test nDCG[@10, @20]: [0.0467, 0.0552]
```

You can also train the model by running 'main.py'.
There are 6 arguments you can control:
- path (any string, default is 'run1'): the path to save the trained model and training log.
- dataset ('movies', 'books', or 'electronics')
- model ('mind' or 'comirec'): the backbone model to use.
    * 'mind': MIND from "Chao Li, Zhiyuan Liu, Mengmeng Wu, Yuchi Xu, Huan Zhao, Pipei Huang, Guoliang Kang, Qiwei Chen, Wei Li, and Dik Lun Lee. 2019. Multi-Interest Network with Dynamic Routing for Recommendation at Tmall. In Proceedings of the 28th ACM International Conference on Information and Knowledge Management (CIKM '19). Association for Computing Machinery, New York, NY, USA, 2615–2623. https://doi.org/10.1145/3357384.3357814".
    * 'comirec': ComiRec-SA from "Yukuo Cen, Jianwei Zhang, Xu Zou, Chang Zhou, Hongxia Yang, and Jie Tang. 2020. Controllable Multi-Interest Framework for Recommendation. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD '20). Association for Computing Machinery, New York, NY, USA, 2942–2951. https://doi.org/10.1145/3394486.3403344".
- selection ('hard', 'p' or 'l): the method of selecting interest from multi-interests.
    * 'hard': hard selection from previous methods.
    * 'p': The proposed soft-selection method BaM_p which uses a simple inner-product function for correlation score (Equation (8)).
    * 'l': The proposed soft-selection method BaM_l which uses additional linear layers for correlation score (Equation (9)).
- tau (any number, default is 1): the softness of selection (tau in Equation (7)). Smaller the tau, softer the selection.
- linear_size (any number smaller than hidden_size, default is 16): the output size of linear layer in BaM_l (d' in Equation (9))

For example, you can train the model for Books dataset with BaM_p and tau of 1.1 on ComiRec at 'bam_p' by following code:
```
python src/main.py --path bam_p --dataset books --model comirec --selection p --tau 1.1
```


You can evaluate the trained_model by running 'main.py' with the argument 'test' as True:
```
python src/main.py --path bam_p --dataset books --model comirec --selection p --tau 1.1 --test True
```

## Datasets
Preprocessed data are included in the data directory.
| Dataset | Users | Items | Interactions | Density |
| --- | ---: | ---: | ---: | ---: |
|Movies & TV (movies)| 22,747 | 17,848 | 841,607 | 0.20% |
|Books (books) | 14,905 | 13,642 | 626,702 | 0.31% |
|Electronics (electronics)| 64,142 | 31,142 |  1,475,538 | 0.07% |

The original datasets are available at https://mengtingwan.github.io/data/goodreads.html.
