mpirun --allow-run-as-root --tag-output -np 256 --hostfile ~/efs/hosts_hvd \
	-map-by ppr:4:socket -mca pml ob1 -mca btl ^openib  -mca btl_tcp_if_include ens5 -x NCCL_DEBUG=INFO -x NCCL_SOCKET_IFNAME=ens5 \
	-x MODULE_VERSION_STACK -x XDG_SESSION_ID -x TERM -x SHELL -x SSH_CLIENT -x SSH_TTY -x USER -x LS_COLORS -x LD_LIBRARY_PATH -x LD_LIBRARY_PATH_WITH_DEFAULT_CUDA -x SSH_AUTH_SOCK -x MODULE_VERSION -x MAIL -x PATH -x PWD -x LANG -x MODULEPATH -x LOADEDMODULES -x LD_LIBRARY_PATH_WITHOUT_CUDA -x SHLVL -x HOME -x PYTHONPATH -x LOGNAME -x XDG_DATA_DIRS -x SSH_CONNECTION -x MODULESHOME -x LESSOPEN -x PKG_CONFIG_PATH -x XDG_RUNTIME_DIR -x LESSCLOSE -x BASH_FUNC_module%% -x _ -x _HOROVOD_SECRET_KEY \
	-x NCCL_MIN_NRINGS=8 -x HOROVOD_HIERARCHICAL_ALLREDUCE=1 \
	-x MXNET_SAFE_ACCUMULATION=1 \
	python run_pretraining_hvd.py \
		--model='bert_24_1024_16' \
		--data='/home/ubuntu/mxnet-data/bert-pretraining/datasets/book-corpus/book-corpus-large-split/*.train,/home/ubuntu/mxnet-data/bert-pretraining/datasets/enwiki/enwiki-feb-doc-split/*.train' \
		--data_eval='/home/ubuntu/mxnet-data/bert-pretraining/datasets/book-corpus/book-corpus-large-split/*.dev,/home/ubuntu/mxnet-data/bert-pretraining/datasets/enwiki/enwiki-feb-doc-split/*.dev' \
		--num_steps 25600 --max_seq_length 128 --lr 0.00354 --warmup_ratio 0.1 \
		--batch_size 2250  --max_predictions_per_seq 20 \
		--accumulate 4 --use_avg_len --raw --log_interval 100 --dtype float32\
		--ckpt_dir ./np256_ckpt_dir \
		--ckpt_interval 1000 2>&1
