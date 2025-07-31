"""
base_vision.py

Abstract class definition of a Vision Backbone (Visual Featurizer), with full annotations of class methods, utility
functions, and initialization logic.

We also define the generic TimmViTBackbone class here, providing a default interface for loading any TIMM Vision
Transformer model for feature extraction.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Optional, Protocol, Tuple, Union

from pathlib import Path

import timm
import torch
import torch.nn as nn
import torchvision.transforms.functional as TVF
from PIL.Image import Image
from timm.models.vision_transformer import Block, VisionTransformer
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy, transformer_auto_wrap_policy
from torchvision.transforms import Compose, Resize
from prismatic.models.backbones.vision.patch_cluster import cluster_dpc_knn, cluster_dpc_knn_diff_features, merge_tokens, ClusterMerger
import math

import numpy as np
from scipy.stats import chi2

# === Utility Functions for Monkey-Patching ===
def unpack_tuple(fn: Callable[[Any], Tuple[Any]]) -> Callable[[Any], Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    return wrapper


# === Interface for an Image Transform ===
class ImageTransform(Protocol):
    def __call__(self, img: Image, **kwargs: str) -> Union[torch.Tensor, Dict[str, torch.Tensor]]: ...


# === Custom Torchvision Image Transforms ===
@dataclass
class LetterboxPad:
    padding_fill_value: Tuple[int, int, int]

    def __call__(self, image: Image) -> Image:
        """Given a PIL.Image, pad to square by adding a symmetric border around the height/width."""
        (w, h), max_wh = image.size, max(image.size)
        horizontal_pad, vertical_pad = int((max_wh - w) / 2), int((max_wh - h) / 2)
        padding = (horizontal_pad, vertical_pad, horizontal_pad, vertical_pad)
        return TVF.pad(image, padding, fill=self.padding_fill_value, padding_mode="constant")

class PatchClusterBlock(nn.Module):
    def __init__(self, cluster_center_features: str, token_assignment_features: str, aggregation_features: str, sample_ratio: float, k: int = 5, cluster_merger_fn=merge_tokens) -> None:
        super().__init__()
        self.cluster_center_features = cluster_center_features
        self.token_assignment_features = token_assignment_features
        self.aggregation_features = aggregation_features
        self.sample_ratio = sample_ratio
        self.k = k
        self.cluster_merger_fn = cluster_merger_fn
    
    def forward(self, patch_features):
        
        if self.cluster_center_features == self.token_assignment_features == self.aggregation_features:
            B, N, C = patch_features.shape
            token_weight = patch_features.new_ones(B, N)
        else:
            B, N, C = patch_features[self.cluster_center_features].shape
            token_weight = patch_features[self.cluster_center_features].new_ones(B, N)
        token_weight = token_weight.unsqueeze(2)

        if self.sample_ratio > 1:
            cluster_num = max(math.ceil(self.sample_ratio), 1)
        else:
            cluster_num = max(math.ceil(N * self.sample_ratio), 1)
        k = min(3, max(cluster_num//2, 1)) if self.k > cluster_num else self.k

        if self.cluster_center_features == self.token_assignment_features == self.aggregation_features:
            index_down, idx_cluster = cluster_dpc_knn(patch_features, cluster_num, k, token_mask=None)
            merged_features = self.cluster_merger_fn(patch_features, idx_cluster, cluster_num, token_weight)
        else:
            index_down, idx_cluster = cluster_dpc_knn_diff_features(patch_features[self.cluster_center_features], patch_features[self.token_assignment_features], cluster_num, k, token_mask=None)
            merged_features = self.cluster_merger_fn(patch_features[self.aggregation_features], idx_cluster, cluster_num, token_weight)

        return index_down, merged_features

class PatchCluster(nn.Module):
    def __init__(self, use_cluster: bool, num_scale_reps: int, cluster_rate0: float, cluster_rate1: float, cluster_rate2: float, cluster_center_features: str, token_assignment_features: str, aggregation_features: str, cluster_merger: str='average', dvt_selected_tokens_path: Optional[Path] = None) -> None:
        super().__init__()
        self.use_cluster = use_cluster
        if self.use_cluster:
            assert num_scale_reps <= 3
            self.num_scale_reps = num_scale_reps
            self.cluster_rate0 = cluster_rate0
            self.cluster_rate1 = cluster_rate1
            self.cluster_rate2 = cluster_rate2
            self.dvt_selected_tokens_path = dvt_selected_tokens_path
            ks = [5, 3, 3]

            if dvt_selected_tokens_path is not None:
                self.dvt_selected_tokens = np.load(dvt_selected_tokens_path).astype(np.int32)
                self.example_num = 0

            if cluster_merger == 'average':
                cluster_merger_fn = merge_tokens
            elif cluster_merger == 'attention':
                cluster_merger_fn = ClusterMerger()
            
            for i in range(self.num_scale_reps):
                sample_ratio = getattr(self, f'cluster_rate{i}')
                setattr(self, f'ctm{i}', PatchClusterBlock(cluster_center_features, token_assignment_features, aggregation_features, sample_ratio, ks[i], cluster_merger_fn))  
    
    def forward(self, patch_features):
        if self.use_cluster:
            if self.dvt_selected_tokens_path is not None:
                if np.isin(self.dvt_selected_tokens[self.example_num], [0, 1]).all():
                    token_indices = torch.from_numpy(np.where(self.dvt_selected_tokens[self.example_num])[0]).to(patch_features.device).sort()[0].unsqueeze(0)
                else: 
                    token_indices = torch.from_numpy(self.dvt_selected_tokens[self.example_num]).to(patch_features.device).sort()[0].unsqueeze(0)
                assert patch_features.size(0) == 1
                token_patch_features = patch_features[:,token_indices[0],:]
                self.example_num += 1
                return token_indices, token_patch_features

            cluster_patch_features = []
            cluster_center_indices = []
            for i in range(self.num_scale_reps):
                cluster_block = getattr(self, f'ctm{i}')
                indices, features = cluster_block(patch_features)

                cluster_patch_features.append(features)
                cluster_center_indices.append(indices)
            if self.num_scale_reps == 0:
                return None, None
            cluster_patch_features = torch.cat(cluster_patch_features, dim=1)
            cluster_center_indices = torch.cat(cluster_center_indices, dim=1)
            return cluster_center_indices, cluster_patch_features
        else:
            return None, patch_features


class FastVPatchCluster(nn.Module):
    def __init__(self, num_clusters: int, tokens_per_cluster: int, filter_clusters: bool = False) -> None:
        super().__init__()

        self.num_clusters = num_clusters
        self.tokens_per_cluster = tokens_per_cluster

        self.k = 5
            
        self.filter_clusters = filter_clusters

    def forward(self, patch_features):
        B, N, C = patch_features.shape
        token_weight = patch_features.new_ones(B, N)
        
        token_weight = token_weight.unsqueeze(2)

        cluster_num = self.num_clusters
        k = min(3, max(cluster_num//2, 1)) if self.k > cluster_num else self.k

        index_down, idx_cluster = cluster_dpc_knn(patch_features, cluster_num, k, token_mask=None)
        assert len(index_down) == 1 and len(idx_cluster) == 1
        index_down = index_down[0]
        idx_cluster = idx_cluster[0]
    
        if self.filter_clusters:
            num_filtered = 0
            idx_cluster_grid = idx_cluster.reshape((27,27))
            filtered_idx_cluster = idx_cluster_grid.clone()
            original_indices = torch.arange(27*27, device=idx_cluster.device).reshape(27,27)

            all_chosen_idxs = []
            for cluster_i in range(self.num_clusters):
                cluster_idxs = (idx_cluster_grid == cluster_i).nonzero(as_tuple=False)

                if len(cluster_idxs) <= self.tokens_per_cluster:
                    chosen_idxs = []
                    for idx in cluster_idxs:
                        chosen_idxs.append(original_indices[idx[0], idx[1]])
                    chosen_idxs = torch.stack(chosen_idxs)
                    all_chosen_idxs.append(chosen_idxs)
                    break

                cluster_center = cluster_idxs.float().mean(dim=0)

                # mahalanobis distance
                cov_matrix = torch.cov(cluster_idxs.T)
                inv_cov_matrix = torch.pinverse(cov_matrix.float())
                distances_from_center = (cluster_idxs - cluster_center).float()
                distances_from_center = torch.sqrt(torch.sum(distances_from_center @ inv_cov_matrix * distances_from_center, dim=-1))

                threshold = np.sqrt(chi2.ppf(0.85, df=cluster_idxs.shape[1]))
                outliers = torch.nonzero(distances_from_center > threshold).flatten()
                filtered_indices = cluster_idxs[outliers]
                for idx in filtered_indices:
                    num_filtered += 1
                    filtered_idx_cluster[idx[0], idx[1]] = -1
                
                new_cluster_idxs = (filtered_idx_cluster == cluster_i).nonzero(as_tuple=False)

                assert self.tokens_per_cluster == 5
                mean_center = torch.mean(new_cluster_idxs.float(), dim=0)
                mean_y, mean_x = mean_center[0], mean_center[1]
                distances_to_mean = torch.norm(new_cluster_idxs - mean_center, dim=1)
                center_idx = torch.argmin(distances_to_mean)
                center_point = new_cluster_idxs[center_idx]
                center_point = original_indices[center_point[0], center_point[1]]

                top_candidates = new_cluster_idxs[new_cluster_idxs[:, 0] == new_cluster_idxs[:, 0].min()]
                top_distances = torch.abs(top_candidates[:, 1] - mean_x)
                top_center = top_candidates[torch.argmin(top_distances)]
                top_center = original_indices[top_center[0], top_center[1]]

                bottom_candidates = new_cluster_idxs[new_cluster_idxs[:, 0] == new_cluster_idxs[:, 0].max()]
                bottom_distances = torch.abs(bottom_candidates[:, 1] - mean_x)
                bottom_center = bottom_candidates[torch.argmin(bottom_distances)]
                bottom_center = original_indices[bottom_center[0], bottom_center[1]]

                left_candidates = new_cluster_idxs[new_cluster_idxs[:, 1] == new_cluster_idxs[:, 1].min()]
                left_distances = torch.abs(left_candidates[:, 0] - mean_y)
                left_center = left_candidates[torch.argmin(left_distances)]
                left_center = original_indices[left_center[0], left_center[1]]

                right_candidates = new_cluster_idxs[new_cluster_idxs[:, 1] == new_cluster_idxs[:, 1].max()]
                right_distances = torch.abs(right_candidates[:, 0] - mean_y)
                right_center = right_candidates[torch.argmin(right_distances)]
                right_center = original_indices[right_center[0], right_center[1]]

                chosen_idxs = torch.stack([top_center, left_center, center_point, right_center, bottom_center])

                all_chosen_idxs.append(chosen_idxs)
 
            all_chosen_idxs = torch.unique(torch.cat(all_chosen_idxs))

            return all_chosen_idxs
        else:
            if self.tokens_per_cluster != 1: assert False
            return index_down

# === Abstract Base Class for arbitrary Vision Backbones ===
class VisionBackbone(nn.Module, ABC):
    def __init__(self, vision_backbone_id: str, image_resize_strategy: str, default_image_size: int = 224) -> None:
        super().__init__()
        self.identifier: str = vision_backbone_id
        self.image_resize_strategy: str = image_resize_strategy
        self.default_image_size: int = default_image_size

        # Instance attributes for a Vision Backbone
        self.featurizer: nn.Module = None
        self.image_transform: ImageTransform = None

    def get_image_transform(self) -> ImageTransform:
        return self.image_transform

    @abstractmethod
    def get_fsdp_wrapping_policy(self) -> Callable: ...

    @abstractmethod
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the featurizer given a set of processed images, returning patch/grid features."""
        raise NotImplementedError

    @property
    @abstractmethod
    def default_image_resolution(self) -> Tuple[int, int, int]: ...

    @property
    @abstractmethod
    def embed_dim(self) -> int: ...

    @property
    @abstractmethod
    def num_patches(self) -> int: ...

    @property
    @abstractmethod
    def half_precision_dtype(self) -> torch.dtype: ...


# === Abstract Base Class for Arbitrary TIMM Vision Transformer Backbones ===
class TimmViTBackbone(VisionBackbone, ABC):
    def __init__(
        self,
        vision_backbone_id: str,
        timm_path_or_url: str,
        image_resize_strategy: str,
        default_image_size: int = 224,
        override_act_layer: Optional[str] = None,
    ) -> None:
        super().__init__(vision_backbone_id, image_resize_strategy, default_image_size=default_image_size)
        self.timm_path_or_url = timm_path_or_url
        self.override_act_layer = override_act_layer
        self.dtype = torch.bfloat16

        # Initialize Featurizer (ViT) by downloading from HF / TIMM Hub if necessary
        if self.override_act_layer is None:
            self.featurizer: VisionTransformer = timm.create_model(
                self.timm_path_or_url, pretrained=True, num_classes=0, img_size=self.default_image_size
            )
        else:
            self.featurizer: VisionTransformer = timm.create_model(
                self.timm_path_or_url,
                pretrained=True,
                num_classes=0,
                img_size=self.default_image_size,
                act_layer=self.override_act_layer,
            )
        self.featurizer.eval()

        # Monkey-Patch the `forward()` function of the featurizer to ensure FSDP-compatibility
        #   => Note: By default set `get_intermediate_layers` to return the *SECOND-TO-LAST* layer patches!
        #   => TODO (siddk) Remove after resolution of https://github.com/pytorch/pytorch/issues/109385
        self.featurizer.forward = unpack_tuple(
            partial(self.featurizer.get_intermediate_layers, n={len(self.featurizer.blocks) - 2})
        )

        ########
        # used for pruning methods like VisionZip
        ########
        import types
        import torch.nn.functional as F
        def attn_save_scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
            L, S = query.size(-2), key.size(-2)
            scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
            attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
            if is_causal:
                assert attn_mask is None
                temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
                attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
                attn_bias.to(query.dtype)

            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
                else:
                    attn_bias = attn_mask + attn_bias

            if enable_gqa:
                key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
                value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

            attn_weight = query @ key.transpose(-2, -1) * scale_factor
            attn_weight += attn_bias
            attn_weight = torch.softmax(attn_weight, dim=-1)
            attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
            return attn_weight @ value, attn_weight

        def attn_save_weights_forward(self, x):
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q, k = self.q_norm(q), self.k_norm(k)

            self.metric = k.clone().mean(1)
            if self.fused_attn:
                x, attn = attn_save_scaled_dot_product_attention(
                    q,k,v,
                    dropout_p=self.attn_drop.p if self.training else 0.,
                )
                self.attn_weight = attn
            else:
                q = q * self.scale
                attn = q @ k.transpose(-2, -1)
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                self.attn_weight = attn
                x = attn @ v

            x = x.transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x
        attn_module = self.featurizer.blocks[len(self.featurizer.blocks) - 2].attn # select second to last layer
        attn_module.forward = types.MethodType(attn_save_weights_forward, attn_module)
        ########

        # Validation =>> for now, this class *only* supports TIMM Vision Transformers (but can be extended!)
        assert isinstance(self.featurizer, VisionTransformer), (
            "Featurizer is not a TIMM VisionTransformer; if you would like to support a new visual representation, "
            "file an issue or implement the requisite logic (see `prismatic/models/backbones/vision/base_vision.py`)!"
        )

        # Get Config =>> Note :: Override default image size to ensure correct image transform
        self.data_cfg = timm.data.resolve_model_data_config(self.featurizer)
        self.data_cfg["input_size"] = (3, self.default_image_size, self.default_image_size)

        # Initialize Default Image Transform --> Modified by `self.image_resize_strategy`
        default_image_transform = timm.data.create_transform(**self.data_cfg, is_training=False)

        # Fix =>> SigLIP & IN1K default transforms resize to *larger* than `self.default_image_size` (crops image)!
        if "siglip" in self.timm_path_or_url or "in1k" in self.timm_path_or_url:
            assert isinstance(default_image_transform, Compose), "Unexpected `default_image_transform`!"
            assert isinstance(default_image_transform.transforms[0], Resize)
            default_image_transform = Compose(
                [
                    Resize(self.default_image_size, interpolation=default_image_transform.transforms[0].interpolation),
                    *default_image_transform.transforms[1:],
                ]
            )

        # Switch on `image_resize_strategy`
        if self.image_resize_strategy == "resize-naive":
            assert isinstance(default_image_transform, Compose), "Unexpected `default_image_transform`!"
            assert isinstance(default_image_transform.transforms[0], Resize)

            target_size = (self.default_image_size, self.default_image_size)
            self.image_transform = Compose(
                [
                    Resize(target_size, interpolation=default_image_transform.transforms[0].interpolation),
                    *default_image_transform.transforms[1:],
                ]
            )

        elif self.image_resize_strategy == "resize-crop":
            self.image_transform = default_image_transform

        elif self.image_resize_strategy == "letterbox":
            assert isinstance(default_image_transform, Compose), "Unexpected `default_image_transform`!"
            assert "mean" in self.data_cfg, "TIMM `data_cfg` missing image normalization mean!"

            # Compute Padding Fill Value (rescaled normalization mean if applicable)
            fill = tuple([int(x * 255) for x in self.data_cfg["mean"]])

            # Build New Transform
            self.image_transform = Compose([LetterboxPad(fill), *default_image_transform.transforms])

        else:
            raise ValueError(f"Image Resize Strategy `{self.image_resize_strategy}` is not supported!")

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return a simple FSDP policy that wraps each ViT block and then the _entire_ featurizer."""
        vit_wrap_policy = partial(_module_wrap_policy, module_classes={VisionTransformer})
        transformer_block_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
        return partial(_or_policy, policies=[vit_wrap_policy, transformer_block_policy])

    def forward(self, pixel_values: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Runs transformed image/pixel tensor through vision backbone, returning _all_ patch features."""

        return self.featurizer(pixel_values)

        ######## PRUNING METHODS (e.g. VisionZip, FasterVLM) ########
        # # VisionZip
        # # dominant_num = 216
        # # contextual_num = 40

        # # FasterVLM
        # dominant_num = 216 + 40
        # contextual_num = 0

        # hidden_states = self.featurizer(pixel_values)
        # attn_weights = self.featurizer.blocks[len(self.featurizer.blocks) - 2].attn.attn_weight # B X H X S X S
        # metric = self.featurizer.blocks[len(self.featurizer.blocks) - 2].attn.metric
        # attn_rec = attn_weights.mean(dim=1).mean(dim=1) # B X S
        # topk_indices = attn_rec.topk(dominant_num, dim=1).indices
        # all_indices = topk_indices

        # mask = torch.ones_like(hidden_states[:, :, 0], dtype=torch.bool, device=metric.device).scatter_(1, all_indices, False)
        # dominant_tokens = hidden_states.masked_select(~mask.unsqueeze(-1)).view(hidden_states.shape[0], dominant_num, hidden_states.shape[2])
        
        # if contextual_num > 0:
        #     ## Filter
        #     metric_filtered = metric[mask].view(hidden_states.shape[0], hidden_states.shape[1] - (dominant_num), metric.shape[2])
        #     hidden_states_filtered = hidden_states.masked_select(mask.unsqueeze(-1)).view(hidden_states.shape[0], hidden_states.shape[1] - (dominant_num), hidden_states.shape[2])  
        #     metric_normalized = metric_filtered / metric_filtered.norm(dim=-1, keepdim=True) 

        #     ## Contextual Visual Tokens
        #     step = max(1, metric_normalized.shape[1] // contextual_num)
        #     target_indices = torch.arange(0, metric_normalized.shape[1], step, device=metric_normalized.device)[:contextual_num]
        #     target_tokens = metric_normalized[:, target_indices, :]

        #     tokens_to_merge = metric_normalized[:, ~torch.isin(torch.arange(metric_normalized.shape[1], device=metric_normalized.device), target_indices), :]
        #     similarity = torch.bmm(tokens_to_merge, target_tokens.transpose(1, 2))
        #     assign_one_hot = torch.zeros(tokens_to_merge.shape[0], tokens_to_merge.shape[1], contextual_num, dtype=hidden_states_filtered.dtype, device=metric_normalized.device)
        #     assign_one_hot.scatter_(2, similarity.argmax(dim=2).unsqueeze(-1), 1)
        #     counts = assign_one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)
        #     hidden_to_merge = hidden_states_filtered[:, ~torch.isin(torch.arange(hidden_states_filtered.shape[1], device=hidden_states_filtered.device), target_indices), :]
        #     aggregated_hidden = torch.bmm(assign_one_hot.transpose(1, 2), hidden_to_merge) / counts
        #     target_hidden = hidden_states_filtered[:, target_indices, :]  
            
        #     contextual_tokens = target_hidden + aggregated_hidden

        #     ## Merge with target hidden states and concatenate
        #     hidden_states_save = torch.cat([dominant_tokens, contextual_tokens], dim=1)
        # else:
        #     hidden_states_save = dominant_tokens
        # return hidden_states_save
        ######## PRUNING METHODS ########

    @property
    def default_image_resolution(self) -> Tuple[int, int, int]:
        return self.data_cfg["input_size"]

    @property
    def embed_dim(self) -> int:
        return self.featurizer.embed_dim

    @property
    def num_patches(self) -> int:
        return self.featurizer.patch_embed.num_patches

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return self.dtype