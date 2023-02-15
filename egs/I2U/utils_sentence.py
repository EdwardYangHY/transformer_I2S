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

config_path='../../config_sentence.yml'
with open(config_path, 'r') as yml:
    config = yaml.safe_load(yml)

dir_name = config["i2u"]["dir_name"]

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

def save_checkpoint(metric, data_name, epoch, model, optimizer, bleu4, accuracy_score, is_best, train_ID ,device=None):
    '''
        使用的scheduel， 是否需要存储其变化的lr？
    '''
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'bleu-4': bleu4,
        'accuracy': accuracy_score,
    }
    filename = 'checkpoint_' + data_name + '.pth.tar'
    ### If use GPU, save differently
    if device != None:
        if device.type=='cuda':
            filename = 'checkpoint_' + data_name + '_gpu.pth.tar'
    ###
    torch.save(state, f"../../saved_model/I2U/{dir_name}/{train_ID}/" + filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, f'../../saved_model/I2U/{dir_name}/{train_ID}/{metric}_BEST_' + filename)

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
    best_bleu4 = checkpoint["bleu-4"]
    best_accuracy = checkpoint["accuracy"]
    return model, optimizer, start_epoch, best_bleu4, best_accuracy

# def save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer,
#                     bleu4, is_best, device=None):
#     #changed
#     """
#     Saves model checkpoint.

#     :param data_name: base name of processed dataset
#     :param epoch: epoch number
#     :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
#     :param encoder: encoder model
#     :param decoder: decoder model
#     :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
#     :param decoder_optimizer: optimizer to update decoder's weights
#     :param bleu4: validation BLEU-4 score for this epoch
#     :param is_best: is this checkpoint the best so far?
#     """
#     state = {'epoch': epoch,
#              'epochs_since_improvement': epochs_since_improvement,
#              'bleu-4': bleu4,
#              'encoder': encoder,
#              'decoder': decoder,
#              'encoder_optimizer': encoder_optimizer,
#              'decoder_optimizer': decoder_optimizer}
    
#     filename = 'checkpoint_' + data_name + '.pth.tar'
#     ### If use GPU, save differently
#     if device != None:
#         if device.type=='cuda':
#             filename = 'checkpoint_' + data_name + '_gpu.pth.tar'
#     ###
#     torch.save(state, f"../../model/I2U/{dir_name}/" + filename)
#     # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
#     if is_best:
#         torch.save(state, f'../../model/I2U/{dir_name}/BEST_' + filename)


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

def get_lr_schedule(optimizer, num_warmup_epochs: int = 10, d_model: int = 2048):
    def lr_lambda(current_epoch: int):
        """
        Eq. (3) in [Transformer paper](https://arxiv.org/abs/1706.03762)
        """
        return d_model**(-0.5) * min((current_epoch+1)**(-0.5), (current_epoch+1)*num_warmup_epochs**(-1.5))

    return LambdaLR(optimizer, lr_lambda, verbose=True)

