import os
import json
import pickle

import torch
from PIL import Image


def main():
    root_dir = "sk_vg/"
    save_dir = "data_sk_vg_onestage/"

    for split in ["train", "val", "test"]:
        split_data = json.load(open(os.path.join(root_dir, split + ".json")))
        samples = []
        for sample in split_data:
            top_left = [int(sample["bbox"]["x"] - 1 / 2 * sample["bbox"]["width"]),
                        int(sample["bbox"]["y"] - 1 / 2 * sample["bbox"]["height"])]
            bbox = [top_left[0], top_left[1], sample["bbox"]["width"], sample["bbox"]["height"]]
            sent = sample["ref_exp"] + ". " + sample["knowledge"]

            samples.append([sample["image_name"], None, bbox, sent, None])
        os.makedirs(os.path.join(save_dir), exist_ok=True)
        torch.save(samples, open(os.path.join(save_dir, f"sk_vg_{split}.pth"), "wb"))


if __name__ == '__main__':
    main()
