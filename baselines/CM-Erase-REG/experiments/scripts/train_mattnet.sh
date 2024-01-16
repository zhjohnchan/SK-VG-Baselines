GPU_ID1=$1

IMDB="coco_minus_refer"
ITERS=1250000
TAG="notime"
NET="res101"
DATASET="sk_vg"
SPLITBY="official"
ID=coco+_pretrain

CUDA_VISIBLE_DEVICES=${GPU_ID1} python -u ./tools/train.py \
    --imdb_name ${IMDB} \
    --net_name ${NET} \
    --iters ${ITERS} \
    --tag ${TAG} \
    --dataset ${DATASET} \
    --splitBy ${SPLITBY} \
    --id ${ID} \
    --max_epochs 15 \
    --learning_rate 4e-4 \
    --erase_train 0 \
    --batch_size 16 \
    2>&1 | tee logs/${ID}
