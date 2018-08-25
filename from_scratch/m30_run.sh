CUDA_VISIBLE_DEVICES=3 \
sh ../scripts/train_from_scratch.py \
    --dset_dir ../data/m30 \
    --num_class 30 \
    --max_epoch 256 \
    --lr 1e-3 \
    --log_dir 30_log
