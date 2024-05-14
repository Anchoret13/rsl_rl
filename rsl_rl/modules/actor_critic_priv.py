import numpy
import torch
import torch.nn as nn
from torch.distributions import Normal

privileged_obs_info = {
    "input dims": [18],
    "latent dims": [18],
    "hidden dims": [[256, 128]]
}
# TO BE UPDATED: including roller state, environment factor

class ActorCritic_priv(nn.Module):
    is_recurrent = False

    def __init__(self, num_obs,
                 num_privileged_obs,
                 num_obs_history,
                 num_actions,
                 **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super().__init__()

        activation = get_activation('elu')
        for i, (branch_input_dim, branch_hidden_dims, branch_latent_dim) in enumerate(
            zip(
                privileged_obs_info["input dims"],
                privileged_obs_info["latent dims"],
                privileged_obs_info["hidden dims"]
            )
        ):
            env_factor_encoder_layers = []
            env_factor_encoder_layers.append(nn.Linear(branch_input_dim, branch_hidden_dims[0]))
            env_factor_encoder_layers.append(activation)
            for l in range(len(branch_hidden_dims)):
                if l == len(branch_hidden_dims) - 1:
                    env_factor_encoder_layers.append(
                        nn.Linear(branch_hidden_dims[l], branch_latent_dim))
                else:
                    env_factor_encoder_layers.append(
                        nn.Linear(branch_hidden_dims[l],
                                  branch_hidden_dims[l + 1]))
                    env_factor_encoder_layers.append(activation)
        self.env_factor_encoder = nn.Sequential(*env_factor_encoder_layers)
        self.add_module(f"encoder", self.env_factor_encoder)
        


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None