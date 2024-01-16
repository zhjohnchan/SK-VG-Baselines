export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch \
 --master_port 6789 \
 --nproc_per_node=4 \
 --use_env train.py \
 --num_workers 4 \
 --use_knowledge \
 --batch_size 16 \
 --lr_bert 0.00001 \
 --aug_crop --aug_scale \
 --aug_translate --backbone resnet50 \
 --detr_model ./checkpoints/detr-r50-referit.pth \
 --bert_enc_num 12 --detr_enc_num 6 --dataset skvg \
 --max_query_len 128 --output_dir outputs/skvg
