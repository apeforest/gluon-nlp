MXNET_SAFE_ACCUMULATION=1 \
python run_pretraining.py \
	--gpus='0,1,2,3,4,5,6,7' \
	--model='bert_24_1024_16' \
	--data='/home/ubuntu/mxnet-data/bert-pretraining/datasets/book-corpus/book-corpus-large-split/*.train,/home/ubuntu/mxnet-data/bert-pretraining/datasets/enwiki/enwiki-feb-doc-split/*.train' \
	--data_eval='/home/ubuntu/mxnet-data/bert-pretraining/datasets/book-corpus/book-corpus-large-split/*.dev,/home/ubuntu/mxnet-data/bert-pretraining/datasets/enwiki/enwiki-feb-doc-split/*.dev' \
	--raw --num_steps 900000 --max_seq_length 128 --max_predictions_per_seq 20 --short_seq_prob 1.0 \
	--lr 1e-4  --warmup_ratio 0.01 --batch_size 32 --batch_size_eval 32 --ckpt_dir ./baseline_ckpt_dir 2>&1
