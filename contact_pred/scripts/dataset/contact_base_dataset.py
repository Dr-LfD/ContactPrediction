from typing import Dict

import torch
import torch.nn
from contact_pred.scripts.model.common.normalizer import LinearNormalizer


class ContactBaseDataset(torch.utils.data.Dataset):
    def get_validation_dataset(self):
        # return an empty dataset by default
        return ContactBaseDataset()

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        raise NotImplementedError()
    
    def __len__(self) -> int:
        return 0
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        output:
            obs: 
                key: T, *
            next_keypose: T, Da
        """
        raise NotImplementedError()