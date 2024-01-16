GPU_ID=$1

DATASET=sk_vg
SPLITBY=official
# IMDB="coco_minus_refer"
# ITERS=1150000
# TAG="notime"
# NET="res101"
ID="coco+_erase"
# ID="mrcn_dets_cmr_with_st"

case ${DATASET} in
    sk_vg)
        for SPLIT in val test
        do
            CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/eval_dets.py \
                --dataset ${DATASET} \
                --splitBy ${SPLITBY} \
                --split ${SPLIT} \
                --id ${ID}
        done
    ;;
    refcoco)
        for SPLIT in val testA testB
        do
            CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/eval_dets.py \
                --dataset ${DATASET} \
                --splitBy ${SPLITBY} \
                --split ${SPLIT} \
                --id ${ID}
        done
    ;;
    refcoco+)
        for SPLIT in val testA testB
        do
            CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/eval_dets.py \
                --dataset ${DATASET} \
                --splitBy ${SPLITBY} \
                --split ${SPLIT} \
                --id ${ID}
        done
    ;;
    refcocog)
        for SPLIT in val test
        do
            CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/eval_dets.py \
                --dataset ${DATASET} \
                --splitBy ${SPLITBY} \
                --split ${SPLIT} \
                --id ${ID}
        done
    ;;
esac