# HRKD: Hierarchical Relational Knowledge Distillation for Cross-domain Language Model Compression
This repository contains the code for the paper in EMNLP 2021: ["HRKD: Hierarchical Relational Knowledge Distillation for Cross-domain Language Model Compression"](https://arxiv.org/abs/2110.08551).


## Requirements
```shell
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```


## Download checkpoints
Download the vocabulary file of BERT-base (uncased) from [HERE](https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt), and put it into `./pretrained_ckpt/`.  
Download the pre-trained checkpoint of BERT-base (uncased) from [HERE](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin), and put it into `./pretrained_ckpt/`.  
Download the 2nd general distillation checkpoint of TinyBERT from [HERE](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT), and extract them into `./pretrained_ckpt/`.


## Prepare dataset
Download the GLUE dataset (containing MNLI) using the script in [HERE](https://github.com/nyu-mll/GLUE-baselines/blob/master/download_glue_data.py), and put the files into `./dataset/glue/`.
Download the Amazon Reviews dataset from [HERE](https://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html), and extract it into `./dataset/amazon_review/`


## Train the teacher model (BERT$_{\rm B}$-single) from single-domain
```shell
bash train_domain.sh
```


## Distill the student model (BERT$_{\rm S}$) with TinyBERT-KD from single-domain
```shell
bash finetune_domain.sh
```


## Train the teacher model (HRKD-teacher) from multi-domain
```shell
bash train_multi_domain.sh
```
And then put the checkpoints to the specified directories (see the beginning of `finetune_multi_domain.py` for more details).


## Distill the student model (BERT$_{\rm S}$) with our HRKD from multi-domain
```shell
bash finetune_multi_domain.sh
```


## Reference
If you find this code helpful for your research, please cite the following paper.
```BibTex
@inproceedings{dong2021hrkd,
  title     = {{HRKD}: Hierarchical Relational Knowledge Distillation for Cross-domain Language Model Compression},
  author    = {Chenhe Dong and Yaliang Li and Ying Shen and Minghui Qiu},
  booktitle = {Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2021}
}
```
