LAMB_BULK=30 EPS_AFTER_SQRT=1 NUMSTEPS=900000 DTYPE=float16 BS=256 ACC=2 MODEL=bert_24_1024_16 MAX_SEQ_LENGTH=128 MAX_PREDICTIONS_PER_SEQ=20 LR=0.0001 LOGINTERVAL=10 CKPTDIR=ckpt_stage1 CKPTINTERVAL=300000 OPTIMIZER=lamb bash hvd.sh
