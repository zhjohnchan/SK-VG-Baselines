import copy
import os
import json
import shutil


def main():
    save_dir = "SK-VG.v1"

    root_dir = "sk_vg/"
    for split in ["train", "val", "test"]:
        split_data = json.load(open(os.path.join(root_dir, split + ".json")))
        for sample in split_data:
            if not os.path.exists(os.path.join(root_dir, sample["image_name"])):
                print(sample["image_name"])

    id2img = {}
    img2id = {}
    img_id = 0
    for split in ["train", "val", "test"]:
        split_data = json.load(open(os.path.join(root_dir, split + ".json")))
        for sample in split_data:
            path = os.path.join(root_dir, sample["image_name"])
            if path not in img2id:
                id2img[img_id] = path
                img2id[path] = img_id
                img_id += 1

    new_data = {"train": [], "val": [], "test": []}
    os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)
    for split in ["train", "val", "test"]:
        split_data = json.load(open(os.path.join(root_dir, split + ".json")))
        for sample in split_data:
            path = os.path.join(root_dir, sample["image_name"])
            tgt_path = os.path.join(save_dir, "images", str(img2id[path]) + path[-4:])
            shutil.copy(path, tgt_path)
            new_sample = copy.deepcopy(sample)
            new_sample["image_name"] = str(img2id[path]) + path[-4:]
            if split == "val":
                if "level" in new_sample:
                    new_sample.pop("level")
            assert len(new_sample.keys()) == 4 or split == "test"
            new_data[split].append(new_sample)
    json.dump(new_data, open(os.path.join(save_dir, "annotations.json"), "wt"), ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
