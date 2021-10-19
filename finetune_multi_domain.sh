#!/bin/bash

NUM_GPU=4
GPU_DEVICES='0,1,2,3'

SEQ_LEN=128
BS=8
EPOCHS=10
LR=5e-5
DISTIL_RATIO=1
HIDDEN_RATIO=1
PRED_RATIO=1
VAL_FREQ=50
SEED=42

TEACHER_MODEL='meta_bert'
STUDENT_MODEL='meta_tinybert'
MNLI_DIR='./dataset/glue'
AMAZON_REVIEW_DIR='./dataset/amazon_review'
VOCAB_PATH='./pretrained_ckpt/bert-base-uncased-vocab.txt'
STUDENT_PRETRAIN_PATH='./pretrained_ckpt/2nd_General_TinyBERT_4L_312D/pytorch_model.bin'



# ---------------- Train ---------------- #

# MNLI
TRAIN_RATIO=0.9
DEV_RATIO=0.1
EXP_DIR='./exp/finetune_multi_domain/mnli/'
TASK='mnli'
DOMAIN='telephone'
bash dist_finetune_multi_domain.sh $NUM_GPU --gpu_devices $GPU_DEVICES --lowercase --seed $SEED --use_graph --hierarchical --teacher_model $TEACHER_MODEL --student_model $STUDENT_MODEL --task $TASK --domain $DOMAIN --val_freq $VAL_FREQ --lr $LR --batch_size $BS --total_epochs $EPOCHS --train_ratio $TRAIN_RATIO --dev_ratio $DEV_RATIO --data_dir $MNLI_DIR --vocab_path $VOCAB_PATH --max_seq_len $SEQ_LEN --distil_ratio $DISTIL_RATIO --hidden_ratio $HIDDEN_RATIO --pred_ratio $PRED_RATIO --student_pretrain_path $STUDENT_PRETRAIN_PATH --exp_dir $EXP_DIR


# Amazon review
EXP_DIR='./exp/finetune_multi_domain/amazon_review/'
TASK='amazon_review'
DOMAIN='books'
bash dist_finetune_multi_domain.sh $NUM_GPU --gpu_devices $GPU_DEVICES --lowercase --seed $SEED --use_graph --hierarchical --teacher_model $TEACHER_MODEL --student_model $STUDENT_MODEL --task $TASK --domain $DOMAIN --val_freq $VAL_FREQ --lr $LR --batch_size $BS --total_epochs $EPOCHS --data_dir $AMAZON_REVIEW_DIR --vocab_path $VOCAB_PATH --max_seq_len $SEQ_LEN --distil_ratio $DISTIL_RATIO --hidden_ratio $HIDDEN_RATIO --pred_ratio $PRED_RATIO --student_pretrain_path $STUDENT_PRETRAIN_PATH --exp_dir $EXP_DIR




# ---------------- Test ---------------- #

# MNLI
GPU_DEVICES='2,3'
EXP_DIR='./exp/finetune_multi_domain/mnli/'
TASK='mnli'

DOMAIN='fiction'
RESUME_PATH=$EXP_DIR'best_model_fiction.bin'
python finetune_multi_domain.py --gpu_devices $GPU_DEVICES --do_test --lowercase --teacher_model $TEACHER_MODEL --student_model $STUDENT_MODEL --task $TASK --domain $DOMAIN --val_freq $VAL_FREQ --lr $LR --batch_size $BS --total_epochs $EPOCHS --data_dir $MNLI_DIR --vocab_path $VOCAB_PATH --max_seq_len $SEQ_LEN --distil_ratio $DISTIL_RATIO --hidden_ratio $HIDDEN_RATIO --pred_ratio $PRED_RATIO --student_pretrain_path $STUDENT_PRETRAIN_PATH --student_resume_path $RESUME_PATH --exp_dir $EXP_DIR

DOMAIN='government'
RESUME_PATH=$EXP_DIR'best_model_government.bin'
python finetune_multi_domain.py --gpu_devices $GPU_DEVICES --do_test --lowercase --teacher_model $TEACHER_MODEL --student_model $STUDENT_MODEL --task $TASK --domain $DOMAIN --val_freq $VAL_FREQ --lr $LR --batch_size $BS --total_epochs $EPOCHS --data_dir $MNLI_DIR --vocab_path $VOCAB_PATH --max_seq_len $SEQ_LEN --distil_ratio $DISTIL_RATIO --hidden_ratio $HIDDEN_RATIO --pred_ratio $PRED_RATIO --student_pretrain_path $STUDENT_PRETRAIN_PATH --student_resume_path $RESUME_PATH --exp_dir $EXP_DIR

DOMAIN='slate'
RESUME_PATH=$EXP_DIR'best_model_slate.bin'
python finetune_multi_domain.py --gpu_devices $GPU_DEVICES --do_test --lowercase --teacher_model $TEACHER_MODEL --student_model $STUDENT_MODEL --task $TASK --domain $DOMAIN --val_freq $VAL_FREQ --lr $LR --batch_size $BS --total_epochs $EPOCHS --data_dir $MNLI_DIR --vocab_path $VOCAB_PATH --max_seq_len $SEQ_LEN --distil_ratio $DISTIL_RATIO --hidden_ratio $HIDDEN_RATIO --pred_ratio $PRED_RATIO --student_pretrain_path $STUDENT_PRETRAIN_PATH --student_resume_path $RESUME_PATH --exp_dir $EXP_DIR

DOMAIN='telephone'
RESUME_PATH=$EXP_DIR'best_model_telephone.bin'
python finetune_multi_domain.py --gpu_devices $GPU_DEVICES --do_test --lowercase --teacher_model $TEACHER_MODEL --student_model $STUDENT_MODEL --task $TASK --domain $DOMAIN --val_freq $VAL_FREQ --lr $LR --batch_size $BS --total_epochs $EPOCHS --data_dir $MNLI_DIR --vocab_path $VOCAB_PATH --max_seq_len $SEQ_LEN --distil_ratio $DISTIL_RATIO --hidden_ratio $HIDDEN_RATIO --pred_ratio $PRED_RATIO --student_pretrain_path $STUDENT_PRETRAIN_PATH --student_resume_path $RESUME_PATH --exp_dir $EXP_DIR

DOMAIN='travel'
RESUME_PATH=$EXP_DIR'best_model_travel.bin'
python finetune_multi_domain.py --gpu_devices $GPU_DEVICES --do_test --lowercase --teacher_model $TEACHER_MODEL --student_model $STUDENT_MODEL --task $TASK --domain $DOMAIN --val_freq $VAL_FREQ --lr $LR --batch_size $BS --total_epochs $EPOCHS --data_dir $MNLI_DIR --vocab_path $VOCAB_PATH --max_seq_len $SEQ_LEN --distil_ratio $DISTIL_RATIO --hidden_ratio $HIDDEN_RATIO --pred_ratio $PRED_RATIO --student_pretrain_path $STUDENT_PRETRAIN_PATH --student_resume_path $RESUME_PATH --exp_dir $EXP_DIR


# Amazon review
GPU_DEVICES='0,1'
EXP_DIR='./exp/finetune_multi_domain/amazon_review/'
TASK='amazon_review'

DOMAIN='books'
RESUME_PATH=$EXP_DIR'best_model_books.bin'
python finetune_multi_domain.py --gpu_devices $GPU_DEVICES --do_test --lowercase --teacher_model $TEACHER_MODEL --student_model $STUDENT_MODEL --task $TASK --domain $DOMAIN --val_freq $VAL_FREQ --lr $LR --batch_size $BS --total_epochs $EPOCHS --data_dir $AMAZON_REVIEW_DIR --vocab_path $VOCAB_PATH --max_seq_len $SEQ_LEN --distil_ratio $DISTIL_RATIO --hidden_ratio $HIDDEN_RATIO --pred_ratio $PRED_RATIO --student_pretrain_path $STUDENT_PRETRAIN_PATH --student_resume_path $RESUME_PATH --exp_dir $EXP_DIR

DOMAIN='dvd'
RESUME_PATH=$EXP_DIR'best_model_dvd.bin'
python finetune_multi_domain.py --gpu_devices $GPU_DEVICES --do_test --lowercase --teacher_model $TEACHER_MODEL --student_model $STUDENT_MODEL --task $TASK --domain $DOMAIN --val_freq $VAL_FREQ --lr $LR --batch_size $BS --total_epochs $EPOCHS --data_dir $AMAZON_REVIEW_DIR --vocab_path $VOCAB_PATH --max_seq_len $SEQ_LEN --distil_ratio $DISTIL_RATIO --hidden_ratio $HIDDEN_RATIO --pred_ratio $PRED_RATIO --student_pretrain_path $STUDENT_PRETRAIN_PATH --student_resume_path $RESUME_PATH --exp_dir $EXP_DIR

DOMAIN='electronics'
RESUME_PATH=$EXP_DIR'best_model_electronics.bin'
python finetune_multi_domain.py --gpu_devices $GPU_DEVICES --do_test --lowercase --teacher_model $TEACHER_MODEL --student_model $STUDENT_MODEL --task $TASK --domain $DOMAIN --val_freq $VAL_FREQ --lr $LR --batch_size $BS --total_epochs $EPOCHS --data_dir $AMAZON_REVIEW_DIR --vocab_path $VOCAB_PATH --max_seq_len $SEQ_LEN --distil_ratio $DISTIL_RATIO --hidden_ratio $HIDDEN_RATIO --pred_ratio $PRED_RATIO --student_pretrain_path $STUDENT_PRETRAIN_PATH --student_resume_path $RESUME_PATH --exp_dir $EXP_DIR

DOMAIN='kitchen'
RESUME_PATH=$EXP_DIR'best_model_kitchen.bin'
python finetune_multi_domain.py --gpu_devices $GPU_DEVICES --do_test --lowercase --teacher_model $TEACHER_MODEL --student_model $STUDENT_MODEL --task $TASK --domain $DOMAIN --val_freq $VAL_FREQ --lr $LR --batch_size $BS --total_epochs $EPOCHS --data_dir $AMAZON_REVIEW_DIR --vocab_path $VOCAB_PATH --max_seq_len $SEQ_LEN --distil_ratio $DISTIL_RATIO --hidden_ratio $HIDDEN_RATIO --pred_ratio $PRED_RATIO --student_pretrain_path $STUDENT_PRETRAIN_PATH --student_resume_path $RESUME_PATH --exp_dir $EXP_DIR
