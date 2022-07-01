python finetune.py --split scaffold --epochs 0 --filename ${dataset}/${gnn_type}_nopretrain --gnn_type $gnn_type --dataset $dataset

python finetune.py --model_file INPUT_MODEL_PATH --dataset DOWNSTREAM_DATASET --filename OUTPUT_FILE_PATH