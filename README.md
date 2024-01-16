# Baselines for SK-VG

## Preprocess Data


For MAttNet, CMErase-REG, NMTree, please use the [refer](https://github.com/lichengunc/refer) format. You can use the processing script:
```shell
python convert_to_refer_format.py
```


For Onestage-Grounding, ReSC, please the [Onestage-Grounding](https://github.com/zyang-ur/onestage_grounding) format. You can use the prcessing script:
```shell
python convert_to_onestage_format.py
```

## [My_MAttNet](https://github.com/lichengunc/MAttNet)

```angular2html
# Put the sk-vg data on the data folder
cd pyutils/refer-parser2/pyutils/refer
make

cd ../..
# Put the sk-vg data on the data folder
# Put stanford-corenlp-full-2013-06-20 on the stanford-corenlp-full-2013-06-20 folder
python parse_sents.py --dataset sk_vg --splitBy official --num_workers 32
python parse_atts.py --dataset sk_vg --splitBy official

cd ../..
python tools/prepro.py --dataset sk_vg --splitBy official
# output: cache/prepro/data.json, cache/prepro/data.h5

cd pyutils/pytorch-faster-rcnn/
# Put the sk-vg data on the data folder
# Put output/res101/coco_2014_train_minus_refer_valtest+coco_2014_valminusminival/notime/res101_mask_rcnn_iter_1250000.pth
# Put data/coco/annotations/instances_minival2014.json

cd ../..
python tools/extract_mrcn_head_feats.py --dataset sk_vg --splitBy official
# output: cache/feats/refcoco_unc/mrcn/res101_coco_minus_refer_notime/xxx.h5

python tools/extract_mrcn_ann_feats.py --dataset sk_vg --splitBy official
# output: cache/feats/refcoco_unc/mrcn/res101_coco_minus_refer_notime_ann_feats.h5

python tools/run_detect.py --dataset sk_vg --splitBy official --conf_thresh 0.65
python tools/extract_mrcn_det_feats.py --dataset sk_vg --splitBy official

./experiments/scripts/train_mattnet.sh GPU_ID sk_vg official

./experiments/scripts/eval_easy.sh GPU_ID sk_vg official
./experiments/scripts/eval_dets.sh GPU_ID sk_vg official
```

## [CM-Erase-REG](https://github.com/xh-liu/CM-Erase-REG)

```angular2html
./experiments/scripts/train_mattnet.sh GPU_ID
./experiments/scripts/train_erase.sh GPU_ID
./experiments/scripts/eval_easy.sh GPU_ID
./experiments/scripts/eval_dets.sh GPU_ID
```

## [NMTree](https://github.com/daqingliu/NMTree)

```angular2html
python tools/train.py \
--id det_nmtree_01 \
--dataset sk_vg \
--split_by official \
--grounding_model NMTree \
--data_file data_dep \
--batch_size 128 \
--glove glove.840B.300d_dep \
--visual_feat_file sk_vg_official_matt_gt_feats.pth

python tools/eval_det.py \
--id det_nmtree_01 \
--log_path log/ \
--dataset sk_vg \
--split_by official \
--visual_feat_file sk_vg_official_matt_det_feats.pth

```

## [Onestage-Grounding](https://github.com/zyang-ur/onestage_grounding)

```angular2html
python train_yolo.py --data_root ./ln_data/ --dataset sk_vg \
--gpu 5 --batch_size 64 --resume saved_models/lstm_referit_model.pth.tar \
--lr 1e-4 --nb_epoch 100 --lstm

python train_yolo.py --data_root ./ln_data/ --dataset sk_vg \
--gpu 5 --resume saved_models/model_sk_vg_batch64_model_best.pth.tar \
--lstm --test

```

## [ReSC](https://github.com/zyang-ur/ReSC)

```angular2html
python train.py --data_root ./ln_data/ --dataset sk_vg \
--gpu 2 --batch_size 16 --resume saved_models/ReSC_base_referit.pth.tar

python train.py --data_root ./ln_data/ --dataset sk_vg \
--gpu 2 --batch_size 16 --resume saved_models/filmconv_nofpn32_sk_vg_batch16_model_best.pth.tar --test
```

## [KeViLI](baselines/TransVG)
```shell
bash train.sh
bash test.sh
```

## [LeViLM](https://github.com/microsoft/GLIP#pre-training)
Follow the instruction to finetune the model.
