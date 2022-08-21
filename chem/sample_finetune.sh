python finetune.py --epochs 0 --gnn_type gcn --input_model_file ./model_architecture/gcn_supervised.pth --dataset tox21 --filename tox21/gcn_debug0 --eval_train

python finetune.py --epochs 1 --gnn_type gcn --input_model_file ./model_architecture/gcn_supervised.pth --dataset tox21 --filename tox21/gcn_debug1 --eval_train

python finetune.py --gnn_type gcn --input_model_file ./model_architecture/gcn_supervised.pth --dataset tox21 --filename tox21/gcn_feature_prompt --eval_train

python finetune.py --gnn_type gcn --input_model_file ./model_architecture/gcn_supervised.pth --dataset hiv --filename hiv/gcn_feature_prompt --eval_train

python finetune.py --gnn_type gcn --input_model_file ./model_architecture/gcn_supervised.pth --dataset tox21 --filename tox21/gcn_finetune -prompt none --eval_train

python finetune.py --gnn_type gcn --dataset tox21 --filename tox21/gcn_nopretrain -prompt none --eval_train

python finetune.py --gnn_type gcn --dataset tox21 --filename tox21/gcn_feature_prompt_nopretrain

python finetune.py --gnn_type gcn --input_model_file ./model_architecture/gcn_supervised.pth --dataset tox21 --filename tox21/gcn_debug --eval_train
python finetune.py --gnn_type gcn --input_model_file ./model_architecture/gcn_supervised.pth --dataset tox21 --filename tox21/gcn_debug -prompt none --eval_train

python finetune.py --gnn_type gcn --input_model_file ./model_architecture/gcn_supervised.pth --dataset tox21 --filename tox21/gcn_debug --eval_train --JK none


python finetune.py --gnn_type gcn --input_model_file ./model_architecture/gcn_supervised.pth --dataset tox21 --filename tox21/gcn_debug --JK none
python finetune.py --gnn_type gcn --input_model_file ./model_architecture/gcn_supervised.pth --dataset hiv --filename hiv/gcn_debug --JK none
