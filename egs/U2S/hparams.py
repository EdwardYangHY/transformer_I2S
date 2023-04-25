import tensorflow as tf
# from text import symbols
import yaml
with open('../../config.yml', 'r') as yml:
    config = yaml.safe_load(yml)

# with open('/net/papilio/storage2/yhaoyuan/transformer_I2S/config.yml', 'r') as yml:
#     config = yaml.safe_load(yml)

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def create_hparams():
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = AttrDict({
        ################################
        # Experiment Parameters        #
        ################################
        "epochs":10000, # 5000， 500
        "iters_per_checkpoint":1000, # 5000
        "seed":1234, # 6789
        "dynamic_loss_scaling":True,
        "fp16_run":False, # True
        "distributed_run":False, # True
        "dist_backend":"nccl",
        "dist_url":"tcp://localhost:54321",
        "cudnn_enabled":True,
        "cudnn_benchmark":False,
        "ignore_layers":['embedding.weight'],

        ################################
        # Data Parameters             #
        ################################
        "load_mel_from_disk":False,
        # training_files='filelists/ljs_audio_text_train_filelist.txt',
        "training_files":config["u2s"]["filelists_train"],
        # validation_files='filelists/ljs_audio_text_val_filelist.txt',
        "validation_files":config["u2s"]["filelists_val"],

        "text_cleaners":['english_cleaners'],

        ################################
        # Audio Parameters             #
        ################################
        "max_wav_value":32768.0,
        # "sampling_rate":22050,
        "sampling_rate":24000, #22050
        "filter_length":1024,
        "hop_length":256,
        "win_length":1024,
        "n_mel_channels":80,
        "mel_fmin":0.0,
        "mel_fmax":8000.0,

        ################################
        # Model Parameters             #
        ################################
        # n_symbols=len(symbols),
        "n_symbols":1024, # 102
        "symbols_embedding_dim":512,

        # Encoder parameters
        "encoder_kernel_size":5,
        "encoder_n_convolutions":3,
        "encoder_embedding_dim":512,

        # Decoder parameters
        "n_frames_per_step":1,  # currently only 1 is supported
        "decoder_rnn_dim":1024,
        "prenet_dim":256,
        "max_decoder_steps":1000, # 1000
        "gate_threshold":0.5,
        "p_attention_dropout":0.1,
        "p_decoder_dropout":0.1,

        # Attention parameters
        "attention_rnn_dim":1024,
        "attention_dim":128,

        # Location Layer parameters
        "attention_location_n_filters":32,
        "attention_location_kernel_size":31,

        # Mel-post processing network parameters
        "postnet_embedding_dim":512,
        "postnet_kernel_size":5,
        "postnet_n_convolutions":5,

        ################################
        # Optimization Hyperparameters #
        ################################
        "use_saved_learning_rate":False,
        "learning_rate":1e-3,
        "weight_decay":1e-6,
        "grad_clip_thresh":1.0,
        "batch_size":config["u2s"]["batch_size"], # 256
        "mask_padding":True  # set model's padded outputs to padded values
    })

    return hparams

if __name__ == "__main__":
    hparams = create_hparams()
    print(hparams)
