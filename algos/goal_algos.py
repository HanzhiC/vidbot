import numpy as np
import copy

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from models.goal import GoalPredictor
import diffuser_utils.dataset_utils as DatasetUtils
from diffuser_utils.guidance_params import COMMON_ACTIONS
import pandas as pd


class GoalPredictorModule(pl.LightningModule):
    def __init__(self, algo_config):
        super(GoalPredictorModule, self).__init__()
        self.algo_config = algo_config
        self.nets = nn.ModuleDict()
        policy_kwargs = algo_config.model

        self.nets["policy"] = GoalPredictor(**policy_kwargs)

    @torch.no_grad()
    def encode_action(self, data_batch, clip_model, max_length=20):
        action_tokens, action_feature = DatasetUtils.encode_text_clip(
            clip_model,
            [data_batch["action_text"]],
            max_length=max_length,
            device="cuda",
        )

        action_tokens.to(self.device)
        action_feature.to(self.device)

        action_text = data_batch["action_text"]
        verb_text = action_text.split(" ")[0]
        if verb_text not in COMMON_ACTIONS:
            verb_text = "other"
        else:
            verb_text = verb_text.replace("-", "")
        verb_text = [verb_text]

        verb_tokens, verb_feature = DatasetUtils.encode_text_clip(
            clip_model,
            verb_text,
            max_length=max_length,
            device="cuda",
        )

        verb_tokens.to(self.device)
        verb_feature.to(self.device)

        data_batch.update({"action_feature": action_feature.float()})
        data_batch.update({"verb_feature": verb_feature.float()})

    def forward(self, data_batch, training=False):
        # self.encode_action(data_batch)
        curr_policy = self.nets["policy"]
        outputs = curr_policy(data_batch, training)
        return outputs
