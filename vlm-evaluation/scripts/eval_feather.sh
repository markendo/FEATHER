model_name="prism-siglip-controlled+7b"

datasets=(
    "ocid-ref-full"
    "refcoco-full"
)

# for all datasets
# datasets=(
#     "gqa-full"
#     "vqa-v2-full"
#     "vizwiz-full"
#     "pope-full"
#     "tally-qa-full"
#     "vsr-full"
#     "ai2d-full"
#     "ocid-ref-full"
#     "refcoco-full"
#     "text-vqa-full"
# )

for dataset in "${datasets[@]}"; do
    echo "Running $model_name on $dataset"

    python scripts/evaluate.py --model_id $model_name --dataset.type $dataset --dataset.root_dir $DATASET_ROOT_DIR --fastv True --fastv_k "[8, 16]" --fastv_ratio 0.7 --fastv_predefined_mask "no_rope_s=3" --save_name "${model_name}_feather_k=8_16_r=0.7_s=3"
    python scripts/score.py --dataset.type $dataset --dataset.root_dir $DATASET_ROOT_DIR --save_name "${model_name}_feather_k=8_16_r=0.7_s=3"
done