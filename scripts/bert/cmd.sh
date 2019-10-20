#TRUNCATE_NORM=1 LAMB_BULK=30 EPS_AFTER_SQRT=1 NUMSTEPS=900000 DTYPE=float16 BS=256 ACC=2 MODEL=bert_24_1024_16 MAX_SEQ_LENGTH=128 MAX_PREDICTIONS_PER_SEQ=20 LR=0.0001 LOGINTERVAL=1 CKPTDIR=ckpt_stage1_adam_1x_kv CKPTINTERVAL=300000 OPTIMIZER=bertadam WARMUP_RATIO=0.0001 bash kvstore.sh

export HOST=hosts_8
export OTHER_HOST=hosts_7
export DOCKER_IMAGE=haibinlin/worker_mxnet:c5fd6fc-1.5-cu90-4590c0-4590c0
export PORT=12448
export NP=64
export NCCLMINNRINGS=1
export TRUNCATE_NORM=1
export LAMB_BULK=30
export EPS_AFTER_SQRT=1
export DTYPE=float16
export MODEL=bert_24_1024_16
export LOGINTERVAL=50
export CKPTDIR=/bert/ckpt_stage1_32k
export CKPTINTERVAL=300000000
export OPTIMIZER=lamb2
export COMMIT=488e4ca
export CLUSHUSER=ubuntu

bash clush-hvd.sh

export LOGINTERVAL=1
export OPTIONS="--synthetic_data\ --eval_use_npz\ --verbose"
NUMSTEPS=10 BS=32768 ACC=2 MAX_SEQ_LENGTH=128 MAX_PREDICTIONS_PER_SEQ=20 LR=0.005 WARMUP_RATIO=0.2 bash mul-hvd.sh

#NUMSTEPS=15625 BS=32768 ACC=8 MAX_SEQ_LENGTH=128 MAX_PREDICTIONS_PER_SEQ=20 LR=0.005 WARMUP_RATIO=0.2 bash mul-hvd.sh
