import torch
from torch import nn
import torchvision
from torchvision.models.resnet import resnet50
from vision_transformer import VisionTransformer
import torchvision.transforms as transforms
from ST_backbone import SwinTransformer
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DinoResEncoder(nn.Module):
    def __init__(self, encoded_image_size=14, embed_dim=2048):
        super(DinoResEncoder, self).__init__()
        self.enc_image_size = encoded_image_size

        # resnet = resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # resnet = resnet50(weights=None)
        # resnet = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')

        #resnet = resnet50(pretrained=False) # pretrained will be removed in higher version 
        resnet = resnet50(weights=None)
        resnet.fc = torch.nn.Identity()
        resnet.load_state_dict(torch.load("../../saved_model/dino_resnet50_pretrain.pth"))

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.to_embedding = nn.Linear(2048, embed_dim)
        self.init_weights()
        self.fine_tune()
    
    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution.
        """
        self.to_embedding.weight.data.uniform_(-0.1, 0.1)

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        batch_size = out.size(0)
        now_embedding_dim = out.size(-1)
        out = out.view(batch_size, -1 , now_embedding_dim)
        out = self.to_embedding(out)
        gx = out.mean(1)
        return out, gx

    def fine_tune(self, fine_tune=False):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class DinoResEncoder_NoPool(nn.Module):
    def __init__(self):
        super(DinoResEncoder_NoPool, self).__init__()
        resnet = resnet50(weights=None)
        resnet.fc = torch.nn.Identity()
        resnet.load_state_dict(torch.load("../../saved_model/dino_resnet50_pretrain.pth"))

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        batch_size = out.size(0)
        now_embedding_dim = out.size(-1)
        out = out.view(batch_size, -1 , now_embedding_dim)
        gx = out.mean(1)
        return out, gx

    def fine_tune(self, fine_tune=False):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune
    
# class DinoResEncoder_FixPooling(nn.Module):
#     def __init__(self, embed_dim=2048):
#         super(DinoResEncoder_FixPooling, self).__init__()
#         #self.enc_image_size = encoded_image_size

#         # resnet = resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

#         # resnet = resnet50(weights=None)
#         # resnet = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')

#         #resnet = resnet50(pretrained=False) # pretrained will be removed in higher version 
#         resnet = resnet50(weights=None)
#         resnet.fc = torch.nn.Identity()
#         resnet.load_state_dict(torch.load("../../saved_model/dino_resnet50_pretrain.pth"))

#         # Remove linear and pool layers (since we're not doing classification)
#         modules = list(resnet.children())[:-2]
#         self.resnet = nn.Sequential(*modules)

#         # Resize image to fixed size to allow input images of variable size
#         self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
#         self.to_embedding = nn.Linear(2048, embed_dim)
#         self.init_weights()
#         self.fine_tune()
    
#     def init_weights(self):
#         """
#         Initializes some parameters with values from the uniform distribution.
#         """
#         self.to_embedding.weight.data.uniform_(-0.1, 0.1)

#     def forward(self, images):
#         """
#         Forward propagation.

#         :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
#         :return: encoded images
#         """
        
#         out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
#         out = self.adaptive_pool(out)  # (batch_size, 2048, 7, 7)
#         out = out.permute(0, 2, 3, 1)  # (batch_size, image_size/32, image_size/32, 2048)
#         batch_size = out.size(0)
#         now_embedding_dim = out.size(-1)
#         out = out.view(batch_size, -1 , now_embedding_dim)
#         out = self.to_embedding(out)
#         gx = out.mean(1)
#         return out, gx

#     def fine_tune(self, fine_tune=False):
#         """
#         Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

#         :param fine_tune: Allow?
#         """
#         for p in self.resnet.parameters():
#             p.requires_grad = False
#         # If fine-tuning, only fine-tune convolutional blocks 2 through 4
#         for c in list(self.resnet.children())[5:]:
#             for p in c.parameters():
#                 p.requires_grad = fine_tune

class ViTEncoder(nn.Module):
    '''
        Only take image size as [224, 224].
        Please try to resize the image before training.
    '''
    def __init__(self, patch_size = 16, qkv_bias=True, embed_dim=2048):
        super(ViTEncoder, self).__init__()
        self.vit = VisionTransformer(img_size=[224],patch_size=patch_size, qkv_bias=qkv_bias)
        # state_dict = torch.hub.load_state_dict_from_url(
        #     url="https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth",
        #     map_location="cuda",
        # )
        # self.vit.load_state_dict(state_dict)
        self.vit.load_state_dict(torch.load("../../saved_model/dino_vitbase16_pretrain.pth"))
        self.vit.eval()
        self.to_embedding = nn.Linear(768, embed_dim)
        self.init_weights()

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution.
        """
        self.to_embedding.weight.data.uniform_(-0.1, 0.1)
    def forward(self, images):
        H, W = images.size(2), images.size(3)
        if H != 224:
            resize = transforms.Resize([224,224])
            images = resize(images)
        out = self.vit(images)  # (batch_size, 1 + 14*14, 768)
        out = self.to_embedding(out) # (batch_size, 1 + 14*14, embedding_dim)
        return out[:,1:], out[:,0]


class STEncoder(nn.Module):
    '''
        Only take image size as [224, 224].
        Please try to resize the image before training.
    '''
    def __init__(self,):
        super(STEncoder, self).__init__()
        self.backbone = SwinTransformer(
            img_size=224, 
            embed_dim=192, 
            depths=[2, 2, 18, 2],
            num_heads=[6, 12, 24, 48],
            window_size=12,
            num_classes=1000
        )
        # How to load frm swin-base-patch4-window7-224?
        self.backbone.from_pretrained()
        self.backbone.load_state_dict(torch.load("../../model/dino_vitbase16_pretrain.pth"))
        self.backbone.eval()


# from datasets import *
# #import torchvision.transforms as transforms

# import yaml
# D_encoder = DinoResEncoder()
# V_encoder = ViTEncoder()

# with open('../../config.yml', 'r') as yml:
#     config = yaml.safe_load(yml)
# dir_name = config["i2u"]["dir_name"]

# # Data parameters
# # data_folder = '/media/ssd/caption data'  # folder with data files saved by create_input_files.py
# data_folder = f'../../data/I2U/processed/{dir_name}/'  # folder with data files saved by create_input_files.py
# # data_name = 'coco_4_cap_per_img_5_min_word_freq'  # base name shared by data files
# #data_name = f'coco_{str(config["i2u"]["captions_per_image"])}_cap_per_img_{str(config["i2u"]["min_word_freq"])}_min_word_freq'  # base name shared by data files
# data_name = f'coco_{str(config["i2u"]["captions_per_image"])}_cap_per_img_{str(config["i2u"]["min_word_freq"])}_min_word_freq'  # base name shared by data files

# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                     std=[0.229, 0.224, 0.225])
# batch_size = 64
# workers = 1  # for data-loading; right now, only 1 works with h5py
# train_loader = torch.utils.data.DataLoader(
#     CaptionDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose([normalize])),
#     batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)


# for i, (imgs, caps, caplens) in enumerate(train_loader):
#     D_imgs = D_encoder(imgs)
#     V_imgs = V_encoder(imgs)
    
#     print(caps)