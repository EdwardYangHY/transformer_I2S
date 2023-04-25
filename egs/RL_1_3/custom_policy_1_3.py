import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F
from stable_baselines3 import TD3
from stable_baselines3.common.policies import BasePolicy, BaseModel
from stable_baselines3.common.torch_layers import get_actor_critic_arch, BaseFeaturesExtractor, NatureCNN
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.td3.policies import TD3Policy
from stable_baselines3.common.utils import safe_mean
from torchvision.models.resnet import resnet50


class ResNet50(th.nn.Module):
    def __init__(self):
        super().__init__()
        # from torchvision.models.resnet import resnet50
        self.cnn = resnet50(pretrained=False)
        self.cnn.fc = th.nn.Identity()
        state_dict = th.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth",
            map_location="cuda",
        )
        self.cnn.load_state_dict(state_dict, strict=False)
        self.cnn.eval()
    
    def forward(self, x):
        x = self.cnn.conv1(x)
        x = self.cnn.bn1(x)
        x = self.cnn.relu(x)
        x = self.cnn.maxpool(x)

        x = self.cnn.layer1(x)
        x = self.cnn.layer2(x)
        x = self.cnn.layer3(x)
        x = self.cnn.layer4(x)
        
        x = x.permute(0, 2, 3, 1)
        batch, _, _, channels = x.size()
        x = x.view(batch, -1, channels)
        return x

class ResNet50_new(th.nn.Module):
    def __init__(self):
        super(ResNet50_new, self).__init__()
        resnet = resnet50(weights=None)
        resnet.fc = th.nn.Identity()
        resnet.load_state_dict(th.load("../../saved_model/dino_resnet50_pretrain.pth"))

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7,7))
        self.fine_tune()
        self.resnet.eval()

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        batch, _, _, channels = out.size()
        out = out.view(batch, -1, channels)
        return out
    
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


class CustomFeaturesExtractorCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 50):
        super().__init__(observation_space, features_dim)

        self.cnn = ResNet50()
        self.state_net = nn.Linear(observation_space.spaces["state"].shape[0], self.features_dim)
        self.fc = nn.Linear(2048, self.features_dim)
        self.relu = nn.ReLU()
    
    def forward(self, observation: Dict[str, th.Tensor]):
        state = self.state_net(observation["state"])
        
        with th.no_grad():
            self.cnn.eval()
            leftimage_features = self.cnn(observation["leftimage"])  # (batch, 49, 2048)
            rightimage_features = self.cnn(observation["rightimage"])  # (batch, 49, 2048)

            flatten_leftimage_features = leftimage_features.reshape(-1, 49*2048)
            flatten_rightimage_features = rightimage_features.reshape(-1, 49*2048)
            leftmean_features = th.mean(leftimage_features, dim=1)
            rightmean_features = th.mean(rightimage_features, dim=1)
        
        if self.features_dim == 2048:
            # Identical to original image features, no need use fc
            features = th.cat((state, leftmean_features, rightmean_features), dim=1)
        else:
            features = th.cat((state, self.fc(leftmean_features), self.fc(rightmean_features)), dim=1)
            features = self.relu(features)
        return features, flatten_leftimage_features, flatten_rightimage_features

class CustomFeaturesExtractorCNN_new(CustomFeaturesExtractorCNN):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 50):
        super().__init__(observation_space, features_dim)

        # self.cnn = ResNet50()
        self.cnn = ResNet50_new()
        """
            how about load adaptive pooling from other model?
        """

        
class CustomActor(BasePolicy):
    """
    Actor network (policy) for TD3.
    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        soft_image_features: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=False,
        )

        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn

        # action_dim = get_action_dim(self.action_space)
        # actor_net = create_mlp(features_dim, action_dim, net_arch, activation_fn, squash_output=True)
        # Deterministic action
        # self.mu = nn.Sequential(*actor_net)
        self.mu = CustomMu(net_arch, activation_fn, soft_image_features)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data

    def forward(self, obs: th.Tensor) -> th.Tensor:
        # assert deterministic, 'The TD3 actor only outputs deterministic actions'
        features = self.extract_features(obs)
        return self.mu(features)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Note: the deterministic deterministic parameter is ignored in the case of TD3.
        #   Predictions are always deterministic.
        return self(observation)


class CustomMu(nn.Module):
    def __init__(self, net_arch: List[List[int]], activation_fn: Type[nn.Module], soft_image_features: bool = True):
        super().__init__()
        assert len(net_arch) == 2 and len(net_arch[0]) == len(net_arch[1]) == 3 and net_arch[0][0] == net_arch[1][0] and net_arch[0][2] == 2
        self.fc1 = nn.Linear(net_arch[0][0], net_arch[0][1])
        self.fc2 = nn.Linear(net_arch[0][1], net_arch[0][2])
        self.fc3 = nn.Linear(net_arch[1][0], net_arch[1][1])
        self.fc4 = nn.Linear(net_arch[1][1], net_arch[1][2], bias=False)
        self.activation_fn = activation_fn()
        self.soft_image_features = soft_image_features # True
        self.use_embed = False
        self.layernorm = nn.LayerNorm(net_arch[1][2])

    def forward(self, x):
        features, leftimage_features, rightimage_features = x

        alpha = self.fc2(self.activation_fn(self.fc1(features)))
        # NOTE: move from line 167
        alpha = F.softmax(alpha, dim=-1)
        
        if self.use_embed:
            embed = self.fc4(self.activation_fn(self.fc3(features)))
            embed = self.layernorm(embed)

        if self.soft_image_features and self.training:
            # alpha = F.softmax(alpha, dim=-1)
            image_features = alpha[:,0:1]*leftimage_features + alpha[:,1:2]*rightimage_features
        else:
            image_features = th.stack([leftimage_features, rightimage_features], dim=1)  
            alpha_unsqueeze = alpha.unsqueeze(-1)  
            index = th.argmax(alpha_unsqueeze, dim=1, keepdim=True) 
            index = index.expand(image_features.size(0), 1, image_features.size(-1)) 
            image_features = th.gather(image_features, dim=1, index=index)  
            image_features = image_features.squeeze(1) 
        
        if self.use_embed:
            action = th.cat([image_features, embed, alpha], dim=1)
        else:
            action = th.cat([image_features, alpha], dim=1)
        return action


class CustomContinuousCriticCNN(BaseModel):
    """
    Critic network(s) for DDPG/SAC/TD3.
    It represents the action-state value function (Q-value function).
    Compared to A2C/PPO critics, this one represents the Q-value
    and takes the continuous action as input. It is concatenated with the state
    and then fed to the network which outputs a single value: Q(s, a).
    For more recent algorithms like SAC/TD3, multiple networks
    are created to give different estimates.
    By default, it creates two critic networks used to reduce overestimation
    thanks to clipped Q-learning (cf TD3 paper).
    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether the features extractor is shared or not
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        # action_dim = get_action_dim(self.action_space)

        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks = []
        for idx in range(n_critics):
            # q_net = create_mlp(features_dim + action_dim, 1, net_arch, activation_fn)
            # q_net = nn.Sequential(*q_net)
            q_net = QNetCNN(net_arch, activation_fn, features_dim)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs)
        qvalue_input = th.cat([features[0], actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        with th.no_grad():
            features = self.extract_features(obs)
        return self.q_networks[0](th.cat([features[0], actions], dim=1))


class QNetCNN(nn.Module):
    def __init__(self, net_arch: List[int], activation_fn: Type[nn.Module], features_dim: int):
        super().__init__()
        # assert len(net_arch) == 3 and net_arch[2] == 1
        assert len(net_arch) == 4 and net_arch[3] == 1
        self.fc1 = nn.Linear(net_arch[0], net_arch[1])
        self.fc2 = nn.Linear(net_arch[1], net_arch[2])

        self.fc3 = nn.Linear(net_arch[2], net_arch[3], bias=False)

        self.activation_fn = activation_fn()
        self.use_embed = False
        self.feature_dim = features_dim

        self.fc_img_feat = nn.Linear(2048, self.feature_dim)
        # self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        f = x[:, :self.feature_dim*3]
        img_feat = x[:, self.feature_dim*3:self.feature_dim*3+49*2048]
        if self.use_embed:
            embed = x[:, self.feature_dim*3+49*2048:-2]

        img_feat = img_feat.view(-1, 49, 2048)
        img_feat = th.mean(img_feat, dim=1)
        if self.feature_dim != 2048:
            img_feat = self.fc_img_feat(img_feat)  # (batch, self.feature_dim*3)

        if self.use_embed:
            x = th.cat([f, img_feat, embed], dim=1)
        else:
            x = th.cat([f, img_feat], dim=1)

        return self.fc3(self.activation_fn(self.fc2(self.activation_fn(self.fc1(x)))))


class CustomTD3PolicyCNN(BasePolicy):
    """
    Policy class (with both actor and critic) for TD3.
    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = CustomFeaturesExtractorCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=False,
        )

        # Default network architecture, from the original paper
        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = [400, 300]

        actor_arch, critic_arch = get_actor_critic_arch(net_arch)

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": actor_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
            # "feature_dim": 
        }
        self.actor_kwargs = self.net_args.copy()
        self.critic_kwargs = self.net_args.copy()
        self.critic_kwargs.update(
            {
                "n_critics": n_critics,
                "net_arch": critic_arch,
                "share_features_extractor": share_features_extractor,
            }
        )

        self.actor, self.actor_target = None, None
        self.critic, self.critic_target = None, None
        self.share_features_extractor = share_features_extractor

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        # Create actor and target
        # the features extractor should not be shared
        self.actor = self.make_actor(features_extractor=None)
        self.actor_target = self.make_actor(features_extractor=None)
        # Initialize the target to have the same weights as the actor
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.actor.optimizer = self.optimizer_class(self.actor.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

        if self.share_features_extractor:
            self.critic = self.make_critic(features_extractor=self.actor.features_extractor)
            # Critic target should not share the features extactor with critic
            # but it can share it with the actor target as actor and critic are sharing
            # the same features_extractor too
            # NOTE: as a result the effective poliak (soft-copy) coefficient for the features extractor
            # will be 2 * tau instead of tau (updated one time with the actor, a second time with the critic)
            self.critic_target = self.make_critic(features_extractor=self.actor_target.features_extractor)
        else:
            # Create new features extractor for each network
            self.critic = self.make_critic(features_extractor=None)
            self.critic_target = self.make_critic(features_extractor=None)

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic.optimizer = self.optimizer_class(self.critic.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

        # Target networks should always be in eval mode
        self.actor_target.set_training_mode(False)
        self.critic_target.set_training_mode(False)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.net_args["activation_fn"],
                n_critics=self.critic_kwargs["n_critics"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
                share_features_extractor=self.share_features_extractor,
            )
        )
        return data

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CustomActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return CustomActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CustomContinuousCriticCNN:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return CustomContinuousCriticCNN(**critic_kwargs).to(self.device)

    def forward(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self._predict(observation, deterministic=deterministic)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Note: the deterministic deterministic parameter is ignored in the case of TD3.
        #   Predictions are always deterministic.
        return self.actor(observation)

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.
        This affects certain modules, such as batch normalisation and dropout.
        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.training = mode


class CustomTD3PolicyCNN_new(CustomTD3PolicyCNN):
    """
    Policy class (with both actor and critic) for TD3.
    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        # features_extractor_class: Type[BaseFeaturesExtractor] = CustomFeaturesExtractorCNN,
        features_extractor_class: Type[BaseFeaturesExtractor] = CustomFeaturesExtractorCNN_new,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
        )

class CustomDDPG(TD3):
    """
    Deep Deterministic Policy Gradient (DDPG).
    Deterministic Policy Gradient: http://proceedings.mlr.press/v32/silver14.pdf
    DDPG Paper: https://arxiv.org/abs/1509.02971
    Introduction to DDPG: https://spinningup.openai.com/en/latest/algorithms/ddpg.html
    Note: we treat DDPG as a special case of its successor TD3.
    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[TD3Policy]],
        env,
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1000,  # 1e6
        learning_starts: int = 0,
        batch_size: int = 64,
        tau: float = 0.005,
        gamma: float = 0,
        train_freq: Union[int, Tuple[int, str]] = (4, "step"),
        gradient_steps: int = 1,
        action_noise = None,
        replay_buffer_class = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            # '''Removed in version 1.7.0'''
            create_eval_env=create_eval_env, 
            # '''
            seed=seed,
            optimize_memory_usage=optimize_memory_usage,
            # Remove all tricks from TD3 to obtain DDPG:
            # we still need to specify target_policy_noise > 0 to avoid errors
            policy_delay=1,
            target_noise_clip=0.0,
            target_policy_noise=0.1,
            _init_setup_model=False,
        )

        # Use only one critic
        if "n_critics" not in self.policy_kwargs:
            self.policy_kwargs["n_critics"] = 1

        if _init_setup_model:
            self._setup_model()

    def learn(
        self,
        total_timesteps: int,
        callback = None,
        log_interval: int = 4,
        eval_env = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "DDPG",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ):

        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            #'''Removed in version 1.7.0'''
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            # '''
            tb_log_name=tb_log_name,
            #'''Removed in version 1.7.0'''
            eval_log_path=eval_log_path,
            #'''
            reset_num_timesteps=reset_num_timesteps,
        )
    def _sample_action(
        self,
        learning_starts: int,
        action_noise = None,
        n_envs: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param n_envs:
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            unscaled_action, _ = self.predict(self._last_obs, deterministic=False)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, gym.spaces.Box):
            # scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                # scaled_action = np.clip(scaled_action + action_noise(), -1, 1)
                unscaled_action = unscaled_action + action_noise().astype(unscaled_action.dtype)

            # We store the scaled action in the buffer
            buffer_action = unscaled_action
            # action = self.policy.unscale_action(scaled_action)
            action = buffer_action
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action
    
    def fix_param(self):
        for p in self.policy.actor.mu.fc1.parameters():
            p.requires_grad = False
        for p in self.policy.actor.mu.fc2.parameters():
            p.requires_grad = False
        for p in self.policy.actor.features_extractor.parameters():
            p.requires_grad = False
        for p in self.policy.critic.features_extractor.parameters():
            p.requires_grad = False
        self.policy.actor.mu.soft_image_features = False
    
    def use_embed(self):
        self.policy.actor.mu.use_embed = True
        self.policy.actor_target.mu.use_embed = True

        self.policy.critic.qf0.use_embed = True
        self.policy.critic_target.qf0.use_embed = True
    
    def disable_soft_features(self):
        self.policy.actor.mu.soft_image_features = False
        self.policy.actor_target.mu.soft_image_features = False
    
    def load_param_from_ddpg(self, model):
        self.policy.actor.features_extractor.load_state_dict(model.policy.actor.features_extractor.state_dict())
        self.policy.critic.features_extractor.load_state_dict(model.policy.critic.features_extractor.state_dict())
        self.policy.actor.mu.fc1.load_state_dict(model.policy.actor.mu.fc1.state_dict())
        self.policy.actor.mu.fc2.load_state_dict(model.policy.actor.mu.fc2.state_dict())
    
    def _dump_logs(self) -> None:
        """
        Write log.
        """
        time_elapsed = time.time() - self.start_time
        fps = int(self.num_timesteps / (time_elapsed + 1e-8))
        self.logger.record("time/episodes", self._episode_num, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if self.use_sde:
            self.logger.record("train/std", (self.actor.get_std()).mean().item())

        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
        # Pass the number of timesteps for tensorboard

        self.logger.record("weight/actor_fc1_weight", th.linalg.matrix_norm(self.policy.actor.mu.fc1.weight))
        self.logger.record("weight/actor_fc1_bias", th.linalg.vector_norm(self.policy.actor.mu.fc1.bias))
        self.logger.record("weight/actor_fc2_weight", th.linalg.matrix_norm(self.policy.actor.mu.fc2.weight))
        self.logger.record("weight/actor_fc2_bias", th.linalg.vector_norm(self.policy.actor.mu.fc2.bias))

        self.logger.record("weight/actor_fc3_weight", th.linalg.matrix_norm(self.policy.actor.mu.fc3.weight))
        self.logger.record("weight/actor_fc3_bias", th.linalg.vector_norm(self.policy.actor.mu.fc3.bias))
        self.logger.record("weight/actor_fc4_weight", th.linalg.matrix_norm(self.policy.actor.mu.fc4.weight))

        self.logger.record("weight/actor_layernorm_weight", th.linalg.vector_norm(self.policy.actor.mu.layernorm.weight))
        self.logger.record("weight/actor_layernorm_bias", th.linalg.vector_norm(self.policy.actor.mu.layernorm.bias))
        
        self.logger.record("weight/critic_fc1_weight", th.linalg.matrix_norm(self.policy.critic.qf0.fc1.weight))
        self.logger.record("weight/critic_fc1_bias", th.linalg.vector_norm(self.policy.critic.qf0.fc1.bias))
        self.logger.record("weight/critic_fc2_weight", th.linalg.matrix_norm(self.policy.critic.qf0.fc2.weight))
        self.logger.record("weight/critic_fc2_bias", th.linalg.vector_norm(self.policy.critic.qf0.fc2.bias))

        self.logger.record("weight/features_extractor_fc_weight", th.linalg.matrix_norm(self.policy.actor.features_extractor.fc.weight))
        self.logger.record("weight/features_extractor_fc_bias", th.linalg.vector_norm(self.policy.actor.features_extractor.fc.bias))
        self.logger.record("weight/features_extractor_state_net_weight", th.linalg.matrix_norm(self.policy.actor.features_extractor.state_net.weight))
        self.logger.record("weight/features_extractor_state_net_bias", th.linalg.vector_norm(self.policy.actor.features_extractor.state_net.bias))
        self.logger.dump(step=self.num_timesteps)
    
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []

        for _ in range(gradient_steps):

            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # Compute the next Q-values: min over all critics targets
                target_q_values = replay_data.rewards

            # Get current Q-values estimates for each critic network
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = sum([F.mse_loss(current_q, target_q_values) for current_q in current_q_values])
            critic_losses.append(critic_loss.item())

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                # Compute actor loss
                actor_loss = -self.critic.q1_forward(replay_data.observations, self.actor(replay_data.observations)).mean()
                actor_losses.append(actor_loss.item())

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))