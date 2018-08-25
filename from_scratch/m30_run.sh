CUDA_VISIBLE_DEVICES=3 \
    python ../scripts/train_from_scratch.py \
    --dset_dir ../data/m30 \
    --num_class 30 \
    --max_epoch 256 \
    --lr 1e-1 \
    --log_dir space_10/30_log
