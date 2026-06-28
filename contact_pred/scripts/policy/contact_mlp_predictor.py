from typing import Dict, Union, Tuple
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from contact_pred.scripts.model.common.normalizer import LinearNormalizer
from contact_pred.scripts.policy.contact_base_predictor import ContactBasePredictor
from contact_pred.scripts.model.vision.dict_image_obs_encoder import DictImageObsEncoder
# from contact_pred.scripts.model.keypose.transformer_for_keypose import TransformerForKeypose
from contact_pred.scripts.model.contact.classification_mlp import Binary_Classification_MLP
from contact_pred.scripts.common.pytorch_util import dict_apply

logger = logging.getLogger(__name__)

class ContactMlpPredictor(ContactBasePredictor):
    def __init__(self,
        shape_meta: dict,
        obs_encoder: DictImageObsEncoder,
        ## model: mlp
        hidden_depth=2,
        hidden_dim=1024,
        dropout=0.1,
    ) -> None:
        super().__init__()

        contact_label_dim = shape_meta['label']['shape'][0]
        obs_feature_dim_dict = obs_encoder.output_shape()

        rgb_keys = obs_encoder.rgb_keys
        low_dim_keys = obs_encoder.low_dim_keys

        # ### -- mode_head: MLP --
        obs_feature_dim = 0
        for key in obs_feature_dim_dict.keys():
            dim = math.prod(obs_feature_dim_dict[key])
            obs_feature_dim += dim

        predictor = Binary_Classification_MLP(
            input_dim=obs_feature_dim,
            output_dim=contact_label_dim,
            hidden_dims=[hidden_dim] * hidden_depth,  # hidden_dims is a list of hidden layer sizes
            dropout=dropout  # dropout can be adjusted as needed
        )

        self.obs_encoder = obs_encoder
        self.predictor = predictor
        self.normalizer = LinearNormalizer()

        # self.obs_feature_dim_dict = obs_feature_dim_dict

    # ========== training ==========
    def get_optimizer(
        self, 
        predictor_weight_decay: float,
        obs_encoder_weight_decay: float,
        learning_rate: float, 
        betas: Tuple[float, float]
    ) -> torch.optim.Optimizer:
        """
        Create optimizer for the three modules:
        2. predictor: MLP
        3. obs_encoder: MultiImageObsEncoder
        """
        optim_groups = []
        optim_groups.append({
            "params": self.predictor.parameters(),
            "weight_decay": predictor_weight_decay  # mode_net does not need weight decay
        })
        optim_groups.append({
            "params": self.obs_encoder.parameters(),
            "weight_decay": obs_encoder_weight_decay
        })
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer
    

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())


    def compute_loss(
        self,
        batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]],
    ) -> torch.Tensor:
        nobs = self.normalizer.normalize(batch["obs"])
        label = batch["label"].detach()

        # encode obs
        nobs_feature = self.obs_encoder(nobs)

        # mode loss
        features = []
        batch_size = label.shape[0]
        for key in nobs_feature.keys():
            feature_item = nobs_feature[key].reshape(batch_size,-1)
            features.append(feature_item)
        nobs_feature_embeddings = torch.cat(features, dim=-1)
        sample_weight = batch.get("loss_weight")
        loss = self.predictor.compute_loss(
            nobs_feature_embeddings,
            label,
            sample_weight=sample_weight,
        )

        return loss

    # ========== inference ==========
    def predict_keypose_and_mode(
        self,
        obs_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        @param obs_dict
            keys = shape_meta["obs"].keys()
        @return tgt_keypose: (B, next_keypose_dim)
        """
        # normalize obs
        nobs = self.normalizer.normalize(obs_dict)
        # encode obs
        nobs_feature = self.obs_encoder(nobs)

        ## predict mode
        features = []
        cam_keys = self.obs_encoder.rgb_keys
        batch_size = nobs[cam_keys[0]].shape[0]
        for key in nobs_feature.keys():
            feature_item = nobs_feature[key].reshape(batch_size,-1)
            features.append(feature_item)
        nobs_feature_embeddings = torch.cat(features, dim=-1)

        pred_label = self.predictor.predict(nobs_feature_embeddings)

        result = {
            "label": pred_label,
        }

        return result
    

    def eval_sampling(self, predictor, sampling_batch, device):

        with torch.no_grad():
            batch = dict_apply(sampling_batch, lambda x: x.to(device, non_blocking=True))
            obs_dict = batch['obs']
            gt_label = batch['label']

            result = predictor.predict_keypose_and_mode(obs_dict)
            pred_label = result['label']

            ratio_label_correct = ((pred_label > 0.5).eq(gt_label > 0.5)).sum().item() / gt_label.shape[0]

            del batch
            del obs_dict
            del result
            del pred_label
            del gt_label

        return ratio_label_correct


