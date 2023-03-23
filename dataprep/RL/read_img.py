import os
import numpy as np
import h5py
import json
import sys
# import torch
#from scipy.misc import imread, imresize # Not supported
from imageio import imread
from PIL import Image
from tqdm import tqdm

is_debug = True if sys.gettrace() else False

def open_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def main():
    img_data_path = "/net/papilio/storage2/yhaoyuan/transformer_I2S/data/food_dataset_VC_shuffle.json"
    with open(img_data_path, "r") as f:
        img_data = json.load(f)
    img_base_path = img_data["image_base_path"]
    img_list_train = [img_data["image_base_path"] + pairdata["image"] for pairdata in img_data["data"]["train"]]
    img_list_eval = [img_data["image_base_path"] + pairdata["image"] for pairdata in img_data["data"]["val"]]
    img_list_test = [img_data["image_base_path"] + pairdata["image"] for pairdata in img_data["data"]["test"]]
    if is_debug:
        img_list_train = img_list_train[:10]
        img_list_eval = img_list_eval[:10]
        img_list_test = img_list_test[:10]
    # train_img_path = "../../data/RL/TRAIN_food_dataset_VC_NAMES.json"
    # val_img_path = "../../data/RL/VAL_food_dataset_VC_NAMES.json"
    # test_img_path = "../../data/RL/TEST_food_dataset_VC_NAMES.json"
    # img_list_train = [img_base_path + "/" + name for name in open_json(train_img_path)]
    # img_list_eval = [img_base_path + "/" + name for name in open_json(val_img_path)]
    # img_list_test = [img_base_path + "/" + name for name in open_json(test_img_path)]
    resolution = 320
    if not os.path.exists(f"../../data/RL/{str(resolution)}"):
        os.mkdir(f"../../data/RL/{str(resolution)}")
    
    for impaths, split in [(img_list_train, "TRAIN"),
                           (img_list_eval, "VAL"),
                           (img_list_test, "TEST")]:
        with h5py.File(f"../../data/RL/{str(resolution)}/{split}_food_dataset_VC_IMAGES_{str(resolution)}.hdf5", "a") as h:
            images = h.create_dataset('images', (len(impaths), 3, resolution, resolution), dtype='uint8')
            # names = h.create_dataset('names', )
            print("\nReading %s images and captions, storing to file...\n" % split)
            names = []
            for i, path in enumerate(tqdm(impaths)):
                img = imread(impaths[i])
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)
                img = np.array(Image.fromarray(img).resize((resolution, resolution)))
                img = img.transpose(2, 0, 1)
                assert img.shape == (3, resolution, resolution)
                assert np.max(img) <= 255

                # Save image to HDF5 file
                images[i] = img
                names.append(path.split("/")[-1])
            assert images.shape[0] == len(names)
            with open(f"../../data/RL/{str(resolution)}/{split}_food_dataset_VC_NAMES.json", "w") as f:
                json.dump(names, f)

if __name__ == "__main__":
    main()