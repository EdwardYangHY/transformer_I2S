import os
import numpy as np
import h5py
import json
import torch
#from scipy.misc import imread, imresize # Not supported
from imageio import imread
from PIL import Image
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample
import pickle
import yaml
import warnings
import math
from torch.optim.lr_scheduler import LambdaLR
import sys

config_path='../../config.yml'
with open(config_path, 'r') as yml:
    config = yaml.safe_load(yml)

dir_name = config["i2u"]["dir_name"]

resolution = config["data"]["image_resolution"]

def create_input_files(dataset, karpathy_json_path, image_folder, captions_per_image, min_word_freq, 
                       output_folder, max_len=100):
    with open(output_folder + 'train_image_paths.pickle', 'rb') as f:
        train_image_paths = pickle.load(f)
    with open(output_folder + 'train_image_captions.pickle', 'rb') as f:
        train_image_captions = pickle.load(f)
    with open(output_folder + 'val_image_paths.pickle', 'rb') as f:
        val_image_paths = pickle.load(f)
    with open(output_folder + 'val_image_captions.pickle', 'rb') as f:
        val_image_captions = pickle.load(f)
    with open(output_folder + 'test_image_paths.pickle', 'rb') as f:
        test_image_paths = pickle.load(f)
    with open(output_folder + 'test_image_captions.pickle', 'rb') as f:
        test_image_captions = pickle.load(f)
    with open(output_folder + 'word_freq.pickle', 'rb') as f:
        word_freq = pickle.load(f)



    # Create word map
    # words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    # word_map = {k: v + 1 for v, k in enumerate(words)}
    # word_map['<unk>'] = len(word_map) + 1
    # word_map['<start>'] = len(word_map) + 1
    # word_map['<end>'] = len(word_map) + 1
    # word_map['<pad>'] = 0

    word_map_path = "../../data/processed/WORDMAP_HUBERT.json"
    # word_map_path = "../../data/processed/WORDMAP_ResDAVEnet.json"
    print(f"Using word map from: {word_map_path}")
    print(f"Resizing Image Resolution: {resolution}")
    print(f"Max token len: {max_len}")
    with open(word_map_path, "r") as f:
        word_map = json.load(f)

    # Create a base/root name for all output files
    base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'

    # Save word map to a JSON
    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)

    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(123)
    for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'), #done
                                   (val_image_paths, val_image_captions, 'VAL'),       #done
                                   (test_image_paths, test_image_captions, 'TEST')]:

        with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
            # Make a note of the number of captions we are sampling per image
            h.attrs['captions_per_image'] = captions_per_image

            # Create dataset inside HDF5 file to store images
            images = h.create_dataset('images', (len(impaths), 3, resolution, resolution), dtype='uint8')

            print("\nReading %s images and captions, storing to file...\n" % os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'))

            enc_captions = []
            caplens = []

            print(f"Model training {captions_per_image} captions per images, given {len(imcaps[0])} captions")

            for i, path in enumerate(tqdm(impaths)):

                # Sample captions
                # imcaps[i] means captions of image i, which can be a lot
                # if imcaps[i] doesn't have enough caps
                
                # make sure that caps in imcaps[i] <= max_len

                hubert_caps_short = [x for x in imcaps[i] if len(x) <= max_len]
                assert len(hubert_caps_short) != 0, f"image {path} has no hubert captions shorter than {max_len}"

                imcaps[i] = hubert_caps_short

                if len(imcaps[i]) < captions_per_image:
                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
                # if imcaps[i] has enough caps
                else:
                    captions = sample(imcaps[i], k=captions_per_image)

                # Sanity check
                assert len(captions) == captions_per_image

                # Read images
                # And reshaping
                img = imread(impaths[i])
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)
                # img = imresize(img, (256, 256))
                # resolution = int(config['data']['image_resolution'])
                img = np.array(Image.fromarray(img).resize((resolution, resolution)))
                img = img.transpose(2, 0, 1)
                assert img.shape == (3, resolution, resolution)
                assert np.max(img) <= 255

                # Save image to HDF5 file
                images[i] = img

                for j, c in enumerate(captions):
                    # Encode captions
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                    # Find caption lengths
                    c_len = len(c) + 2
                    assert c_len <= max_len + 2

                    enc_captions.append(enc_c)
                    caplens.append(c_len)

            # Sanity check
            # images数量 X 每个image的caption == enc_captions的总长度
            assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)

            # Save encoded captions and their lengths to JSON files
            with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
                json.dump(caplens, j)

def init_embedding(embeddings):
    """
    Fills embedding tensor with values from the uniform distribution.

    :param embeddings: embedding tensor
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def load_embeddings(emb_file, word_map):
    """
    Creates an embedding tensor for the specified word map, for loading into the model.

    :param emb_file: file containing embeddings (stored in GloVe format)
    :param word_map: word map
    :return: embeddings in the same order as the words in the word map, dimension of embeddings
    """

    # Find embedding dimension
    with open(emb_file, 'r') as f:
        emb_dim = len(f.readline().split(' ')) - 1

    vocab = set(word_map.keys())

    # Create tensor to hold embeddings, initialize
    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    init_embedding(embeddings)

    # Read embedding file
    print("\nLoading embeddings...")
    for line in open(emb_file, 'r'):
        line = line.split(' ')

        emb_word = line[0]
        embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

        # Ignore word if not in train_vocab
        if emb_word not in vocab:
            continue

        embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

    return embeddings, emb_dim


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def save_checkpoint(data_name, epoch, model, optimizer, metric_name, metric_value, is_best, dir_name, train_ID ,device=None, save_epoch=False):
    '''
        使用的scheduel， 是否需要存储其变化的lr？
    '''
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metric_name': metric_name,
        'metric_value': metric_value,
    }
    filename = 'checkpoint_' + data_name + '.pth.tar'
    ### If use GPU, save differently
    if device != None:
        if device.type=='cuda':
            if not save_epoch:
                filename = 'checkpoint_' + data_name + '_gpu.pth.tar'
            else:
                if epoch == 0 or (epoch+1)%5 == 0:
                    filename = str(epoch)+'_checkpoint_' + data_name + '_gpu.pth.tar'
                else:
                    filename = 'checkpoint_' + data_name + '_gpu.pth.tar'
        else:
            raise NotImplementedError
    
    if device.type=="cuda":
        """
            if save_epoch, then save pth for each epoch;
            else, override all previous models when having a new one
        """ 
        if save_epoch:
            if (epoch+1) % 5 == 0:
                filename = str(epoch)+'_checkpoint_' + data_name + '_gpu.pth.tar'
        else:
            filename = 'checkpoint_' + data_name + '_gpu.pth.tar'
    else:
        raise NotImplementedError
    
    torch.save(state, f"../../saved_model/I2U/{dir_name}/{train_ID}/" + filename)
    
    if is_best:
        """
            if is best, define a unified name for BEST model
        """ 
        filename = f"{metric_name}_BEST_checkpoint_{data_name}_gpu.pth.tar"
        torch.save(state, f"../../saved_model/I2U/{dir_name}/{train_ID}/" + filename)
    ###
    # torch.save(state, f"../../saved_model/I2U/{dir_name}/{train_ID}/" + filename)

def save_checkpoint_LM(data_name, epoch, model, optimizer, metric_name, metric_value, is_best, dir_name, train_ID ,device=None, save_epoch=False):
    '''
        使用的scheduel， 是否需要存储其变化的lr？
    '''
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metric_name': metric_name,
        'metric_value': metric_value,
    }
    filename = 'checkpoint_' + data_name + '.pth.tar'
    ### If use GPU, save differently
    if device != None:
        if device.type=='cuda':
            filename = 'checkpoint_' + data_name + '_gpu.pth.tar'
    ###
    torch.save(state, f"../../saved_model/LM/{dir_name}/{train_ID}/" + filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, f'../../saved_model/LM/{dir_name}/{train_ID}/{metric_name}_BEST_' + filename)

def load_checkpoint(checkpoint_path, model, optimizer, device):
    checkpoint = checkpoint_path
    print(f"Loading checkpoint from {checkpoint}")
    # checkpoint = torch.load(checkpoint)
    checkpoint = torch.load(checkpoint, map_location=device)
    start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)
    metric_name = checkpoint["metric_name"]
    metric_value = checkpoint["metric_value"]
    return model, optimizer, start_epoch, metric_name, metric_value


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

def get_lr_schedule(optimizer, num_warmup_epochs: int = 10, last_epoch: int = 0, d_model: int = 2048):
    
    def lr_lambda(current_epoch: int):
        """
        Eq. (3) in [Transformer paper](https://arxiv.org/abs/1706.03762)
        """
        return d_model**(-0.5) * min((current_epoch+1)**(-0.5), (current_epoch+1)*num_warmup_epochs**(-1.5))

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch, verbose=True)

# def load_tacotron2(model_path, max_decoder_step):
#     hparams = create_hparams()
#     hparams.sampling_rate = 22050
#     hparams.max_decoder_steps = max_decoder_step
#     checkpoint_path = model_path
#     tacotron2_model = load_model(hparams)
#     tacotron2_model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
#     tacotron2_model.cuda().eval()
#     return tacotron2_model

# def load_tacotron2_hubert(model_path, code_dict_path, max_decoder_steps):
#     tacotron_model, sample_rate, hparams = load_tacotron(
#         tacotron_model_path=model_path,
#         max_decoder_steps=max_decoder_steps,
#     )

#     if not os.path.exists(hparams.code_dict):
#         hparams.code_dict = code_dict_path
#     tts_dataset = TacotronInputDataset(hparams)
#     return tacotron_model, tts_dataset

# def load_hifigan(checkpoint_path, device):
#     checkpoint_file = checkpoint_path
#     config_file = os.path.join(os.path.split(checkpoint_file)[0], 'config.json')
#     with open(config_file) as f:
#         data = f.read()
#     # global h
#     json_config = json.loads(data)
#     h = AttrDict(json_config)
#     generator = Generator(h).to(device)
#     assert os.path.isfile(checkpoint_file)
#     checkpoint_dict = torch.load(checkpoint_file, map_location=device)
#     generator.load_state_dict(checkpoint_dict['generator'])
#     generator.eval()
#     generator.remove_weight_norm()
#     return generator

# def load_asr(model_path, device):
#     processor = Wav2Vec2Processor.from_pretrained(config["asr"]["model_path"])
#     asr_model = Wav2Vec2ForCTC.from_pretrained(config["asr"]["model_path"]).to(device)
#     asr_model.eval()
#     return asr_model, processor

# def seq2words(seq, rev_word_map, special_words):
#     return [rev_word_map[ind] for ind in seq if rev_word_map[ind] not in special_words]

# def u2s(words, tacotron2_model, hifigan_model, device):
#     # words = [rev_word_map[ind] for ind in seq if rev_word_map[ind] not in special_words]
#     # print(words)
#     sequence = np.array(text_to_sequence(' '.join(words), ['english_cleaners']))[None, :]
#     sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
#     _, mel_outputs_postnet, _, _ = tacotron2_model.inference(sequence)
#     with torch.no_grad():
#         x = mel_outputs_postnet.squeeze().to(device)
#         y_g_hat = hifigan_model(mel_outputs_postnet)
#         audio = y_g_hat.squeeze()
#         # audio = audio * 32768.0
#         # audio = audio.cpu().numpy().astype('int16')
#         audio = audio.cpu().numpy().astype(np.float64)
#     return audio

# def synthesize_mel(model, inp, lab=None, strength=0.0):
#     assert inp.size(0) == 1
#     inp = inp.cuda()
#     if lab is not None:
#         lab = torch.LongTensor(1).cuda().fill_(lab)

#     with torch.no_grad():
#         _, mel, _, ali, has_eos = model.inference(inp, lab, ret_has_eos=True)
#     return mel, has_eos

# def u2s_hubert(words, tacotron2_model, tts_dataset, hifigan_model, device):
#     quantized_units_str = " ".join(words)
#     tts_input = tts_dataset.get_tensor(quantized_units_str)
#     mel, has_eos = synthesize_mel(
#         tacotron2_model,
#         tts_input.unsqueeze(0),
#     )
#     with torch.no_grad():
#         x = mel.squeeze().float()
#         # x = torch.FloatTensor(x).to(device)
#         y_g_hat = hifigan_model(x)
#         audio = y_g_hat.squeeze()
#         audio = audio * 32768.0
#         # audio = audio.cpu().numpy().astype('int16')
#         audio = audio.cpu().numpy().astype(np.float64)
#     return audio

# def s2t(audio, asr_processor, asr_model, device):
#     input_values = asr_processor(audio, sampling_rate=16000, return_tensors="pt").input_values.float()
#     logits = asr_model(input_values.to(device)).logits
#     predicted_ids = torch.argmax(logits, dim=-1)
#     transcription = asr_processor.decode(predicted_ids[0])
#     return transcription