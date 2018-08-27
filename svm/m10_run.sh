CUDA_VISIBLE_DEVICES=1 \
python3 ../scripts/train_svm.py \
    --dset_dir ../data/m10 \
    --num_class 10 \
    --max_epoch 256 \
    --lr 1e-1 \
    --log_dir space_15/10_log \
    --ckpt_fname /home/lixin/Documents/mygithub/voxelnet-cls/from_scratch/space_15/30_log/best.pth.tar 
