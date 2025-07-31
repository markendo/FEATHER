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

    python scripts/evaluate.py --model_id $model_name --dataset.type $dataset --dataset.root_dir $DATASET_ROOT_DIR --fastv True --fastv_k "[3]" --fastv_ratio 0.75 --fastv_predefined_mask "no_rope" --save_name "${model_name}_fastv_k=3_r=0.75_no_rope"
    python scripts/score.py --dataset.type $dataset --dataset.root_dir $DATASET_ROOT_DIR --save_name "${model_name}_fastv_k=3_r=0.75_no_rope"
done