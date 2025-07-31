"""
load.py

Entry point for loading pretrained VLMs for inference; exposes functions for listing available models (with canonical
IDs, mappings to paper experiments, and short descriptions), as well as for loading models (from disk or HF Hub).
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Union

from huggingface_hub import hf_hub_download

from prismatic.models.materialize import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform, get_patch_cluster, get_fastv_patch_cluster
from prismatic.models.registry import GLOBAL_REGISTRY, MODEL_REGISTRY
from prismatic.models.vlms import PrismaticVLM
from prismatic.overwatch import initialize_overwatch

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# === HF Hub Repository ===
HF_HUB_REPO = "TRI-ML/prismatic-vlms"


# === Available Models ===
def available_model_ids() -> List[str]:
    return list(MODEL_REGISTRY.keys())


def available_model_ids_and_names() -> List[List[str]]:
    return list(GLOBAL_REGISTRY.values())


def get_model_description(model_id_or_name: str) -> str:
    if model_id_or_name not in GLOBAL_REGISTRY:
        raise ValueError(f"Couldn't find `{model_id_or_name = }; check `prismatic.available_model_names()`")

    # Print Description & Return
    print(json.dumps(description := GLOBAL_REGISTRY[model_id_or_name]["description"], indent=2))

    return description


# === Load Pretrained Model ===
def load(
    model_id_or_path: Union[str, Path], hf_token: Optional[str] = None, cache_dir: Optional[Union[str, Path]] = None, fastv: Optional[bool] = False, fastv_k: Optional[List[int]] = 3, fastv_ratio: Optional[float] = 0.5, fastv_predefined_mask: Optional[str] = None, dvt_inference: Optional[bool] = False, dvt_num_tokens: Optional[int] = 256, dvt_selected_tokens_path: Optional[Path] = None,
) -> PrismaticVLM:
    """Loads a pretrained PrismaticVLM from either local disk or the HuggingFace Hub."""
    if os.path.isdir(model_id_or_path):
        overwatch.info(f"Loading from local path `{(run_dir := Path(model_id_or_path))}`")

        # Get paths for `config.json` and pretrained checkpoint
        config_json, checkpoint_pt = run_dir / "config.json", run_dir / "checkpoints" / "latest-checkpoint.pt"
        assert config_json.exists(), f"Missing `config.json` for `{run_dir = }`"
        assert checkpoint_pt.exists(), f"Missing checkpoint for `{run_dir = }`"
    else:
        if model_id_or_path not in GLOBAL_REGISTRY:
            raise ValueError(f"Couldn't find `{model_id_or_path = }; check `prismatic.available_model_names()`")

        overwatch.info(f"Downloading `{(model_id := GLOBAL_REGISTRY[model_id_or_path]['model_id'])} from HF Hub")
        config_json = hf_hub_download(repo_id=HF_HUB_REPO, filename=f"{model_id}/config.json", cache_dir=cache_dir)
        checkpoint_pt = hf_hub_download(
            repo_id=HF_HUB_REPO, filename=f"{model_id}/checkpoints/latest-checkpoint.pt", cache_dir=cache_dir
        )

    # Load Model Config from `config.json`
    with open(config_json, "r") as f:
        model_cfg = json.load(f)["model"]

    # = Load Individual Components necessary for Instantiating a VLM =
    #   =>> Print Minimal Config
    overwatch.info(
        f"Found Config =>> Loading & Freezing [bold blue]{model_cfg['model_id']}[/] with:\n"
        f"             Vision Backbone =>> [bold]{model_cfg['vision_backbone_id']}[/]\n"
        f"             LLM Backbone    =>> [bold]{model_cfg['llm_backbone_id']}[/]\n"
        f"             Arch Specifier  =>> [bold]{model_cfg['arch_specifier']}[/]\n"
        f"             Checkpoint Path =>> [underline]`{checkpoint_pt}`[/]"
    )

    # Load Vision Backbone
    overwatch.info(f"Loading Vision Backbone [bold]{model_cfg['vision_backbone_id']}[/]")
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        model_cfg["vision_backbone_id"],
        model_cfg["image_resize_strategy"],
    )
    
    if fastv and dvt_inference:
        assert model_cfg["llm_backbone_id"].endswith("-pure")
        model_cfg["llm_backbone_id"] = model_cfg["llm_backbone_id"].replace('pure', 'dvt-inference-fastv-pure') # assuming pure
    elif dvt_inference:
        assert model_cfg["llm_backbone_id"].endswith("-pure")
        model_cfg["llm_backbone_id"] = model_cfg["llm_backbone_id"].replace('pure', 'dvt-inference-pure') # assuming pure
    elif fastv:
        assert model_cfg["llm_backbone_id"].endswith("-pure")
        model_cfg["llm_backbone_id"] = model_cfg["llm_backbone_id"].replace('pure', 'fastv-pure') # assuming pure
    # Load LLM Backbone --> note `inference_mode = True` by default when calling `load()`
    overwatch.info(f"Loading Pretrained LLM [bold]{model_cfg['llm_backbone_id']}[/] via HF Transformers")
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        model_cfg["llm_backbone_id"],
        llm_max_length=model_cfg.get("llm_max_length", 2048),
        hf_token=hf_token,
        inference_mode=True,
        fastv=fastv,
        fastv_k=fastv_k,
        fastv_ratio=fastv_ratio,
        fastv_predefined_mask=fastv_predefined_mask,
    )

    if fastv_predefined_mask is not None and 'sample_cluster' in fastv_predefined_mask:
        cluster_info = fastv_predefined_mask[fastv_predefined_mask.index('sample_cluster') + len('sample_cluster_'):]
        num_clusters = int(cluster_info.split('_')[0])
        tokens_per_cluster = int(cluster_info.split('_')[1])
        filter_clusters = False
        if len(cluster_info.split('_')) == 3 and cluster_info.split('_')[2] == 'filter':
            filter_clusters = True
        fastv_patch_cluster = get_fastv_patch_cluster(num_clusters=num_clusters, tokens_per_cluster=tokens_per_cluster, filter_clusters=filter_clusters)
    else:
        fastv_patch_cluster = None

    if dvt_inference:
        num_clusters = dvt_num_tokens
        if num_clusters == 0:
            patch_cluster = get_patch_cluster(use_cluster=True, num_scale_reps=0, cluster_rate0=None, cluster_rate1=None, cluster_rate2=None, cluster_center_features=None, token_assignment_features=None, aggregation_features=None)
        elif dvt_selected_tokens_path is not None:
            print(f'using saved selected tokens from {dvt_selected_tokens_path}')
            patch_cluster = get_patch_cluster(use_cluster=True, num_scale_reps=0, cluster_rate0=None, cluster_rate1=None, cluster_rate2=None, cluster_center_features=None, token_assignment_features=None, aggregation_features=None, dvt_selected_tokens_path=dvt_selected_tokens_path)
        else:
            print(f'{num_clusters} clusters')
            patch_cluster = get_patch_cluster(use_cluster=True, num_scale_reps=1, cluster_rate0=num_clusters, cluster_rate1=0, cluster_rate2=0, cluster_center_features='same', token_assignment_features='same', aggregation_features='same')
    else:
        if "use_cluster" in model_cfg:
            if "cluster_center_features" not in model_cfg: # added after some experiments
                model_cfg["cluster_center_features"] = None
                model_cfg["token_assignment_features"] = None
                model_cfg["aggregation_features"] = None
            patch_cluster = get_patch_cluster(
                model_cfg["use_cluster"], model_cfg["num_scale_reps"], model_cfg["cluster_rate0"], model_cfg["cluster_rate1"], model_cfg["cluster_rate2"], model_cfg["cluster_center_features"], model_cfg["token_assignment_features"], model_cfg["aggregation_features"]
            )
        else:
            patch_cluster = get_patch_cluster(False, 1, 1, 1, 1, None, None, None)

    # Load VLM using `from_pretrained` (clobbers HF syntax... eventually should reconcile)
    overwatch.info(f"Loading VLM [bold blue]{model_cfg['model_id']}[/] from Checkpoint; Freezing Weights 🥶")
    vlm = PrismaticVLM.from_pretrained(
        checkpoint_pt,
        model_cfg["model_id"],
        vision_backbone,
        llm_backbone,
        patch_cluster,
        fastv_patch_cluster,
        arch_specifier=model_cfg["arch_specifier"],
    )

    if dvt_inference:
        vlm.dvt_inference = True
        vlm.dvt_num_tokens = dvt_num_tokens
        
        if dvt_selected_tokens_path is not None:
            vlm.use_saved_selected_tokens = True

    return vlm
