for i in `seq 0 4`
do
    CUDA_VISIBLE_DEVICES=0 \
    python ../scripts/train_fine_tune.py \
        --dset_dir ../data/m10 \
        --num_class 10 \
        --max_epoch 256 \
        --lr 1e-1 \
        --few \
        --log_dir space_15/few_10_log/$i \
        --ckpt_fname /home/lixin/Documents/mygithub/voxelnet-cls/from_scratch/space_15/30_log/best.pth.tar 
done
