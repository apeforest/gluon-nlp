mpirun --allow-run-as-root --tag-output -np 256 --hostfile /home/ubuntu/efs/hosts32 \
	-map-by ppr:4:socket -mca pml ob1 -mca btl ^openib  -mca btl_tcp_if_include ens5 -x NCCL_DEBUG=INFO -x NCCL_SOCKET_IFNAME=ens5 \
	-x NCCL_MIN_NRINGS=8 -x HOROVOD_HIERARCHICAL_ALLREDUCE=1 \
	-x MXNET_SAFE_ACCUMULATION=1 \
	python run_pretraining_hvd.py \
		--model='bert_24_1024_16' \
		--data='/home/ubuntu/mxnet-data/bert-pretraining/datasets/book-corpus/book-corpus-large-split/*.train,/home/ubuntu/mxnet-data/bert-pretraining/datasets/enwiki/enwiki-feb-doc-split/*.train' \
		--data_eval='/home/ubuntu/mxnet-data/bert-pretraining/datasets/book-corpus/book-corpus-large-split/*.dev,/home/ubuntu/mxnet-data/bert-pretraining/datasets/enwiki/enwiki-feb-doc-split/*.dev' \
		--num_steps 25000 --max_seq_length 512 --short_seq_prob 1.0 --lr 1e-4 \
		--warmup_ratio 0.01 \
		--batch_size 4 --max_predictions_per_seq 80 \
		--raw --start_step 1 --ckpt_dir ./np256_ckpt_dir 2>&1
