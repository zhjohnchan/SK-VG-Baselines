import os
import json


def main():
    root_dir = "sk_vg/"
    for split in ["train", "val", "test"]:
        split_data = json.load(open(os.path.join(root_dir, split + ".json")))
        for sample in split_data:
            if not os.path.exists(os.path.join(root_dir, sample["image_name"])):
                print(sample["image_name"])


if __name__ == '__main__':
    main()
