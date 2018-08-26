CUDA_VISIBLE_DEVICES=2 \
python3 ../scripts/train_from_scratch.py \
    --dset_dir ../data/m10 \
    --num_class 10 \
    --max_epoch 256 \
    --lr 1e-1 \
    --log_dir space_10/10_log
