from typing import Union, Dict

import unittest
import zarr
import numpy as np
import torch
import torch.nn as nn
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.dict_of_tensor_mixin import DictOfTensorMixin


class LinearNormalizer(DictOfTensorMixin):
    avaliable_modes = ['limits', 'gaussian']
    
    @torch.no_grad()
    def fit(self,
        data: Union[Dict, torch.Tensor, np.ndarray, zarr.Array],
        last_n_dims=1,
        dtype=torch.float32,
        mode='limits',
        output_max=1.,
        output_min=-1.,
        range_eps=1e-4,
        fit_offset=True,
        ## 新增小方差维度处理
        adjust_small_variance=True,
        small_variance_threshold=0.05,
        min_range_multiplier=4.0,
        target_keys=None,
        verbose=True): # 指定需要调整的key, None表示全部


        if isinstance(data, dict):

            # 分析所有数据的方差分布
            if verbose and adjust_small_variance:
                print("\n=== 数据方差分析 ===")
                for key, value in data.items():
                    if isinstance(value, np.ndarray):
                        tensor_data = torch.from_numpy(value)
                    else:
                        tensor_data = value
                    
                    # reshape for analysis
                    if len(tensor_data.shape) > 2:
                        original_shape = tensor_data.shape
                        if last_n_dims > 0:
                            dim = np.prod(original_shape[-last_n_dims:])
                            tensor_data = tensor_data.reshape(-1, dim)
                        else:
                            tensor_data = tensor_data.reshape(-1, 1)
                    
                    std_per_dim = tensor_data.std(dim=0)
                    small_var_count = (std_per_dim < small_variance_threshold).sum().item()
                    
                    print(f"\n{key}:")
                    print(f"  总维度数: {len(std_per_dim)}")
                    print(f"  小方差维度数: {small_var_count}")
                    print(f"  小方差比例: {small_var_count/len(std_per_dim)*100:.1f}%")
                    # print(f"  std范围: [{std_per_dim.min():.6f}, {std_per_dim.max():.6f}]")
                    
                    if small_var_count > 0:
                        small_var_indices = (std_per_dim < small_variance_threshold).nonzero().squeeze()
                        if small_var_indices.numel() == 1:
                            small_var_indices = [small_var_indices.item()]
                        else:
                            small_var_indices = small_var_indices.tolist()
                        print(f"  小方差维度索引: {small_var_indices}")
            
            ## 正则化数据
            for key, value in data.items():

                # 判断是否需要对当前key进行小方差处理
                should_adjust = adjust_small_variance and \
                                 (target_keys is None or key in target_keys)

                if verbose and should_adjust:
                    print(f"\n=== 处理 {key} (启用小方差调整) ===")
                elif verbose:
                    print(f"\n=== 处理 {key} (未启用小方差调整) ===")
                
                self.params_dict[key] = _fit(value, 
                    last_n_dims=last_n_dims,
                    dtype=dtype,
                    mode=mode,
                    output_max=output_max,
                    output_min=output_min,
                    range_eps=range_eps,
                    fit_offset=fit_offset,
                    ## 新增小方差维度处理
                    adjust_small_variance=should_adjust,
                    small_variance_threshold=small_variance_threshold,
                    min_range_multiplier=min_range_multiplier)


        else:
            self.params_dict['_default'] = _fit(data, 
                last_n_dims=last_n_dims,
                dtype=dtype,
                mode=mode,
                output_max=output_max,
                output_min=output_min,
                range_eps=range_eps,
                fit_offset=fit_offset,
                ## 新增小方差维度处理
                adjust_small_variance=adjust_small_variance,
                small_variance_threshold=small_variance_threshold,
                min_range_multiplier=min_range_multiplier)
            
    
    def __call__(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self.normalize(x)
    
    def __getitem__(self, key: str):
        return SingleFieldLinearNormalizer(self.params_dict[key])

    def __setitem__(self, key: str , value: 'SingleFieldLinearNormalizer'):
        self.params_dict[key] = value.params_dict

    def _normalize_impl(self, x, forward=True):
        if isinstance(x, dict):
            result = dict()
            for key, value in x.items():
                params = self.params_dict[key]
                result[key] = _normalize(value, params, forward=forward)
            return result
        else:
            if '_default' not in self.params_dict:
                raise RuntimeError("Not initialized")
            params = self.params_dict['_default']
            return _normalize(x, params, forward=forward)

    def normalize(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self._normalize_impl(x, forward=True)

    def unnormalize(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self._normalize_impl(x, forward=False)

    def get_input_stats(self) -> Dict:
        if len(self.params_dict) == 0:
            raise RuntimeError("Not initialized")
        if len(self.params_dict) == 1 and '_default' in self.params_dict:
            return self.params_dict['_default']['input_stats']
        
        result = dict()
        for key, value in self.params_dict.items():
            if key != '_default':
                result[key] = value['input_stats']
        return result


    def get_output_stats(self, key='_default'):
        input_stats = self.get_input_stats()
        if 'min' in input_stats:
            # no dict
            return dict_apply(input_stats, self.normalize)
        
        result = dict()
        for key, group in input_stats.items():
            this_dict = dict()
            for name, value in group.items():
                this_dict[name] = self.normalize({key:value})[key]
            result[key] = this_dict
        return result


class SingleFieldLinearNormalizer(DictOfTensorMixin):
    avaliable_modes = ['limits', 'gaussian']
    
    @torch.no_grad()
    def fit(self,
            data: Union[torch.Tensor, np.ndarray, zarr.Array],
            last_n_dims=1,
            dtype=torch.float32,
            mode='limits',
            output_max=1.,
            output_min=-1.,
            range_eps=1e-4,
            fit_offset=True):
        self.params_dict = _fit(data, 
            last_n_dims=last_n_dims,
            dtype=dtype,
            mode=mode,
            output_max=output_max,
            output_min=output_min,
            range_eps=range_eps,
            fit_offset=fit_offset)
    
    @classmethod
    def create_fit(cls, data: Union[torch.Tensor, np.ndarray, zarr.Array], **kwargs):
        obj = cls()
        obj.fit(data, **kwargs)
        return obj
    
    @classmethod
    def create_manual(cls, 
            scale: Union[torch.Tensor, np.ndarray], 
            offset: Union[torch.Tensor, np.ndarray],
            input_stats_dict: Dict[str, Union[torch.Tensor, np.ndarray]]):
        def to_tensor(x):
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x)
            x = x.flatten()
            return x
        
        # check
        for x in [offset] + list(input_stats_dict.values()):
            assert x.shape == scale.shape
            assert x.dtype == scale.dtype
        
        params_dict = nn.ParameterDict({
            'scale': to_tensor(scale),
            'offset': to_tensor(offset),
            'input_stats': nn.ParameterDict(
                dict_apply(input_stats_dict, to_tensor))
        })
        return cls(params_dict)

    @classmethod
    def create_identity(cls, dtype=torch.float32):
        scale = torch.tensor([1], dtype=dtype)
        offset = torch.tensor([0], dtype=dtype)
        input_stats_dict = {
            'min': torch.tensor([-1], dtype=dtype),
            'max': torch.tensor([1], dtype=dtype),
            'mean': torch.tensor([0], dtype=dtype),
            'std': torch.tensor([1], dtype=dtype)
        }
        return cls.create_manual(scale, offset, input_stats_dict)

    def normalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return _normalize(x, self.params_dict, forward=True)

    def unnormalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return _normalize(x, self.params_dict, forward=False)

    def get_input_stats(self):
        return self.params_dict['input_stats']

    def get_output_stats(self):
        return dict_apply(self.params_dict['input_stats'], self.normalize)

    def __call__(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self.normalize(x)



def _fit(data: Union[torch.Tensor, np.ndarray, zarr.Array],
        last_n_dims=1,
        dtype=torch.float32,
        mode='limits',
        output_max=1.,
        output_min=-1.,
        range_eps=1e-4,
        fit_offset=True,
        ## 新增小方差维度处理
        adjust_small_variance=False,
        small_variance_threshold=0.05,
        min_range_multiplier=4.0
        ):
    assert mode in ['limits', 'gaussian']
    assert last_n_dims >= 0
    assert output_max > output_min

    # convert data to torch and type
    if isinstance(data, zarr.Array):
        data = data[:]
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    if dtype is not None:
        data = data.type(dtype)

    # convert shape
    dim = 1
    if last_n_dims > 0:
        dim = np.prod(data.shape[-last_n_dims:])
    data = data.reshape(-1,dim)

    # compute input stats min max mean std
    input_min, _ = data.min(axis=0)
    input_max, _ = data.max(axis=0)
    input_mean = data.mean(axis=0)
    input_std = data.std(axis=0)

    # compute scale and offset
    if mode == 'limits':
        if fit_offset:
            # unit scale
            input_range = input_max - input_min

            ## == 新增：基于方差的小方差维度处理 ==
            if adjust_small_variance:
                ## 识别小方差维度
                small_var_mask = input_std < small_variance_threshold
                ## 对小方差维度进行处理
                if small_var_mask.any():
                    print(f"\n调整 {small_var_mask.nonzero(as_tuple=True)[0]} 维度的小方差维度:")
                    print(f"Original std: {input_std[small_var_mask]}")
                    print(f"Original range: {input_range[small_var_mask]}")

                    ## 基于方差的调整策略：使用标准差的倍数作为有效范围
                    ## 对小方差维度，使用增强的标准差来计算归一化范围
                    enhanced_std = input_std.clone()
                    enhanced_std[small_var_mask] = small_variance_threshold
                    
                    ## 使用增强后的标准差计算有效范围（使用多倍标准差作为范围）
                    variance_based_range = enhanced_std * min_range_multiplier
                    
                    ## 对小方差维度，使用基于方差的范围替代原始范围
                    adjusted_range = input_range.clone()
                    adjusted_range[small_var_mask] = torch.maximum(
                        input_range[small_var_mask],
                        variance_based_range[small_var_mask]
                    )
                    
                    ## 更新输入范围
                    input_range = adjusted_range

                    print(f"Enhanced std: {enhanced_std[small_var_mask]}")
                    print(f"Variance-based range: {variance_based_range[small_var_mask]}")
                    print(f"Final adjusted range: {input_range[small_var_mask]}")
                    
                    ## 计算调整比例
                    adjustment_ratios = adjusted_range[small_var_mask] / (input_max[small_var_mask] - input_min[small_var_mask])
                    print(f"Range expansion ratios: {adjustment_ratios}\n")
            ## == 结束新增小方差维度处理 ==

            ignore_dim = input_range < range_eps
            input_range[ignore_dim] = output_max - output_min
            scale = (output_max - output_min) / input_range
            offset = output_min - scale * input_min
            offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]
            # ignore dims scaled to mean of output max and min

        else:
            # use this when data is pre-zero-centered.
            assert output_max > 0
            assert output_min < 0
            # unit abs
            output_abs = min(abs(output_min), abs(output_max))
            input_abs = torch.maximum(torch.abs(input_min), torch.abs(input_max))

            ## === 新增：基于方差的小方差维度调整 (for fit_offset=False) ===
            if adjust_small_variance:
                small_var_mask = input_std < small_variance_threshold
                if small_var_mask.any():
                    print(f"Detected small variance dimensions (fit_offset=False): {small_var_mask.nonzero(as_tuple=True)[0]}")
                    print(f"Original std: {input_std[small_var_mask]}")
                    print(f"Original input_abs: {input_abs[small_var_mask]}")
                    
                    ## 基于增强的标准差计算绝对值范围
                    enhanced_std = input_std.clone()
                    enhanced_std[small_var_mask] = small_variance_threshold
                    
                    ## 使用增强标准差的倍数作为绝对值范围
                    variance_based_abs = enhanced_std * min_range_multiplier / 2  # 除以2因为是绝对值
                    
                    ## 调整小方差维度的绝对值范围
                    input_abs[small_var_mask] = torch.maximum(
                        input_abs[small_var_mask],
                        variance_based_abs[small_var_mask]
                    )
                    
                    print(f"Enhanced std: {enhanced_std[small_var_mask]}")
                    print(f"Variance-based abs range: {variance_based_abs[small_var_mask]}")
                    print(f"Final adjusted input_abs: {input_abs[small_var_mask]}")
            ## === 小方差调整结束 ===

            ignore_dim = input_abs < range_eps
            input_abs[ignore_dim] = output_abs
            # don't scale constant channels 
            scale = output_abs / input_abs
            offset = torch.zeros_like(input_mean)

    elif mode == 'gaussian':

        # === 新增：基于方差的gaussian模式调整 ===
        if adjust_small_variance:
            ## 对于gaussian模式，直接基于方差阈值调整std
            original_std = input_std.clone()
            small_var_mask = input_std < small_variance_threshold
            
            if small_var_mask.any():
                print(f"Gaussian mode - detected small variance dimensions: {small_var_mask.nonzero().squeeze().tolist()}")
                print(f"Original std: {input_std[small_var_mask]}")
                
                ## 直接将小方差维度的std设置为阈值
                input_std[small_var_mask] = small_variance_threshold
                
                print(f"Adjusted std: {input_std[small_var_mask]}")
                print(f"Std enhancement ratios: {input_std[small_var_mask] / original_std[small_var_mask]}")
        # === 小方差调整结束 ===

        ignore_dim = input_std < range_eps
        scale = input_std.clone()
        scale[ignore_dim] = 1
        scale = 1 / scale

        if fit_offset:
            offset = - input_mean * scale
        else:
            offset = torch.zeros_like(input_mean)
    
    # save
    this_params = nn.ParameterDict({
        'scale': scale,
        'offset': offset,
        'input_stats': nn.ParameterDict({
            'min': input_min,
            'max': input_max,
            'mean': input_mean,
            'std': input_std
        })
    })
    for p in this_params.parameters():
        p.requires_grad_(False)
    return this_params


def _normalize(x, params, forward=True):
    assert 'scale' in params
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    scale = params['scale']
    offset = params['offset']
    x = x.to(device=scale.device, dtype=scale.dtype)
    src_shape = x.shape
    x = x.reshape(-1, scale.shape[0])
    if forward:
        x = x * scale + offset
    else:
        x = (x - offset) / scale
    x = x.reshape(src_shape)
    return x


def test():
    data = torch.zeros((100,10,9,2)).uniform_()
    data[...,0,0] = 0

    normalizer = SingleFieldLinearNormalizer()
    normalizer.fit(data, mode='limits', last_n_dims=2)
    datan = normalizer.normalize(data)
    assert datan.shape == data.shape
    assert np.allclose(datan.max(), 1.)
    assert np.allclose(datan.min(), -1.)
    dataun = normalizer.unnormalize(datan)
    assert torch.allclose(data, dataun, atol=1e-7)

    input_stats = normalizer.get_input_stats()
    output_stats = normalizer.get_output_stats()

    normalizer = SingleFieldLinearNormalizer()
    normalizer.fit(data, mode='limits', last_n_dims=1, fit_offset=False)
    datan = normalizer.normalize(data)
    assert datan.shape == data.shape
    assert np.allclose(datan.max(), 1., atol=1e-3)
    assert np.allclose(datan.min(), 0., atol=1e-3)
    dataun = normalizer.unnormalize(datan)
    assert torch.allclose(data, dataun, atol=1e-7)

    data = torch.zeros((100,10,9,2)).uniform_()
    normalizer = SingleFieldLinearNormalizer()
    normalizer.fit(data, mode='gaussian', last_n_dims=0)
    datan = normalizer.normalize(data)
    assert datan.shape == data.shape
    assert np.allclose(datan.mean(), 0., atol=1e-3)
    assert np.allclose(datan.std(), 1., atol=1e-3)
    dataun = normalizer.unnormalize(datan)
    assert torch.allclose(data, dataun, atol=1e-7)


    # dict
    data = torch.zeros((100,10,9,2)).uniform_()
    data[...,0,0] = 0

    normalizer = LinearNormalizer()
    normalizer.fit(data, mode='limits', last_n_dims=2)
    datan = normalizer.normalize(data)
    assert datan.shape == data.shape
    assert np.allclose(datan.max(), 1.)
    assert np.allclose(datan.min(), -1.)
    dataun = normalizer.unnormalize(datan)
    assert torch.allclose(data, dataun, atol=1e-7)

    input_stats = normalizer.get_input_stats()
    output_stats = normalizer.get_output_stats()

    data = {
        'obs': torch.zeros((1000,128,9,2)).uniform_() * 512,
        'action': torch.zeros((1000,128,2)).uniform_() * 512
    }
    normalizer = LinearNormalizer()
    normalizer.fit(data)
    datan = normalizer.normalize(data)
    dataun = normalizer.unnormalize(datan)
    for key in data:
        assert torch.allclose(data[key], dataun[key], atol=1e-4)
    
    input_stats = normalizer.get_input_stats()
    output_stats = normalizer.get_output_stats()

    state_dict = normalizer.state_dict()
    n = LinearNormalizer()
    n.load_state_dict(state_dict)
    datan = n.normalize(data)
    dataun = n.unnormalize(datan)
    for key in data:
        assert torch.allclose(data[key], dataun[key], atol=1e-4)
