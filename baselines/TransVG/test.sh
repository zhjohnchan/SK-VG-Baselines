export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m torch.distributed.launch \
 --nproc_per_node=1 --use_env eval.py \
 --use_knowledge --batch_size 32 \
 --num_workers 4 --bert_enc_num 12 \
 --detr_enc_num 6 --backbone resnet50 \
 --dataset knowvg-v4 --max_query_len 20 \
 --eval_set test --eval_model outputs/skvg/best_checkpoint.pth \
 --output_dir ./outputs/skvg-test
