import torch
from transformer_I2S.egs.I2U.utils_previous import *       #changed
from models import TransformerLM, TransformerConditionedLM
import pickle
from imageio import imread
from PIL import Image
import torch.nn.functional as F
import yaml
import torchvision.transforms as transforms

# model_name = "trimmed_mapping_SpeakerALL"
# config_path = f"/net/papilio/storage2/yhaoyuan/transformer_I2U/saved_model/I2U/{model_name}/Trial_1/config.yml"
# model_path = f"/net/papilio/storage2/yhaoyuan/transformer_I2U/saved_model/I2U/{model_name}/Trial_1/bleu-4_BEST_checkpoint_coco_2_cap_per_img_1_min_word_freq_gpu.pth.tar"
# test_img = f"/net/papilio/storage2/yhaoyuan/transformer_I2U/data/processed/{model_name}/test_image_paths.pickle"
# test_img_caps = f"/net/papilio/storage2/yhaoyuan/transformer_I2U/data/processed/{model_name}/test_image_captions.pickle"

model_name = "encodec1"
config_path = f"/net/papilio/storage2/yhaoyuan/transformer_I2U/saved_model/I2U/{model_name}/Trial_1/config.yml"
model_path = f"/net/papilio/storage2/yhaoyuan/transformer_I2U/saved_model/I2U/{model_name}/Trial_1/bleu-4_BEST_checkpoint_coco_1_cap_per_img_1_min_word_freq_gpu.pth.tar"
test_img = f"/net/papilio/storage2/yhaoyuan/transformer_I2U/data/processed/{model_name}/val_image_paths.pickle"
test_img_caps = f"/net/papilio/storage2/yhaoyuan/transformer_I2U/data/processed/{model_name}/val_image_captions.pickle"

with open(config_path, 'r') as yml:
    config = yaml.safe_load(yml)

dir_name = config["i2u"]["dir_name"]
model_params = config["i2u"]["model_params"]
train_params = config["i2u"]["train_params"]
data_folder = f'../../data/processed/{dir_name}/'  # folder with data files saved by create_input_files.py
# data_name = 'coco_4_cap_per_img_5_min_word_freq'  # base name shared by data files
#data_name = f'coco_{str(config["i2u"]["captions_per_image"])}_cap_per_img_{str(config["i2u"]["min_word_freq"])}_min_word_freq'  # base name shared by data files
data_name = f'coco_{str(config["i2u"]["captions_per_image"])}_cap_per_img_{str(config["i2u"]["min_word_freq"])}_min_word_freq'  # base name shared by data files

word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
with open(word_map_file, 'r') as j:
    word_map = json.load(j)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
# device = torch.device("cpu")
# Initialize / load checkpoint
model_params['vocab_size'] = len(word_map)
model = TransformerConditionedLM(**model_params)

optimizer = getattr(torch.optim, "Adam")(model.parameters(), lr=train_params["lr"])
checkpoint = model_path
model, optimizer, start_epoch, best_bleu4, best_accuracy = load_checkpoint(checkpoint, model, optimizer, device)
model.eval()
model.to(device)

start_unit = word_map["<start>"]
end_unit = word_map["<end>"]
vocab_size = len(word_map)
max_len = 500



with open(test_img, "rb") as f:
    test_img = pickle.load(f)
with open(test_img_caps, "rb") as f:
    test_img_caps = pickle.load(f)

ref = [word_map[str(unit)] for unit in test_img_caps[10][0]]

def read_img(img_path):
    img = imread(img_path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    # img = imresize(img, (256, 256))
    resolution = int(config['data']['image_resolution'])
    img = np.array(Image.fromarray(img).resize((resolution, resolution)))
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    img = transform(img)  # (3, 256, 256)
    assert img.shape == (3, resolution, resolution)
    # assert np.max(img) <= 255
    return img

img = read_img(img_path=test_img[10])
img = img.unsqueeze(dim = 0)
print(ref)
seqs = model.decode(img, start_unit, end_unit, 500, 100)

print(ref)
print(seqs[1:])
print(img)