from pathlib import Path
from typing import Optional, List

from vlm_eval.util.interfaces import VLM

# from .instructblip import InstructBLIP
# from .llava import LLaVa
from .prismatic import PrismaticVLM

# === Initializer Dispatch by Family ===
#FAMILY2INITIALIZER = {"instruct-blip": InstructBLIP, "llava-v15": LLaVa, "prismatic": PrismaticVLM}
FAMILY2INITIALIZER = {"prismatic": PrismaticVLM}



def load_vlm(
    model_family: str,
    model_id: str,
    run_dir: Path,
    hf_token: Optional[str] = None,
    ocr: Optional[bool] = False,
    load_precision: str = "bf16",
    max_length=128,
    temperature=1.0,
    fastv=False,
    fastv_k: Optional[List[int]] = [3],
    fastv_ratio: Optional[float] = 0.5,
    fastv_predefined_mask: Optional[str] = None,
    dvt_inference=False,
    dvt_num_tokens=None,
    dvt_selected_tokens_path=None,
) -> VLM:
    assert model_family in FAMILY2INITIALIZER, f"Model family `{model_family}` not supported!"
    return FAMILY2INITIALIZER[model_family](
        model_family=model_family,
        model_id=model_id,
        run_dir=run_dir,
        hf_token=hf_token,
        load_precision=load_precision,
        max_length=max_length,
        temperature=temperature,
        ocr=ocr,
        fastv=fastv,
        fastv_k=fastv_k,
        fastv_ratio=fastv_ratio,
        fastv_predefined_mask=fastv_predefined_mask,
        dvt_inference=dvt_inference,
        dvt_num_tokens=dvt_num_tokens,
        dvt_selected_tokens_path=dvt_selected_tokens_path
    )
