CUDA_VISIBLE_DEVICES=1 \
    python3 ../scripts/train_fine_tune.py \
    --dset_dir ../data/m30 \
    --num_class 30 \
    --max_epoch 256 \
    --lr 1e-1 \
    --log_dir space_15/30_log \
    --ckpt_fname /home/lixin/Documents/mygithub/voxelnet-cls/from_scratch/space_15/30_log/best.pth.tar 
