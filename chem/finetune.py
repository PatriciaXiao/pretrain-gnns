import argparse

from loader import MoleculeDataset
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN, GNN_graphpred
from sklearn.metrics import roc_auc_score

from splitters import scaffold_split
import pandas as pd

import os
import shutil

from tensorboardX import SummaryWriter

# from util import ExamineConnectedComponents
from util import PreprocessPrompt

# the debug command
# python finetune.py --gnn_type gcn --input_model_file ./model_architecture/gcn_supervised.pth --dataset tox21 --filename tox21/gcn_debug --eval_train --JK none --dropout_ratio 0.01 --lr 1 --batch_size 2

criterion = nn.BCEWithLogitsLoss(reduction = "none")
#sigmoid = nn.Sigmoid()
#criterion = nn.BCELoss()

def train(args, model, device, loader, optimizer):
    model.train()

    total_loss = 0
    total_step = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        # print(batch.__dict__)

        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.subgraph, batch.prompt_mask)

        # """
        y = batch.y.view(pred.shape).to(torch.float64)

        """debug
        y = 0-batch.y.view(pred.shape).to(torch.float64)
        pred = pred[:,:1]
        y = y[:,:1]
        y[0,0] = -1
        y[1,0] = 1
        # """

        #Whether y is non-null or not.
        is_valid = y**2 > 0
        #Loss matrix
        loss_mat = criterion(pred.double(), (y+1)/2) # original version
        #out_sigm = sigmoid(pred.double())
        #loss_mat = criterion(out_sigm, (y+1)/2)

        #loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            
        optimizer.zero_grad()
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        total_loss += loss.item()
        total_step += 1

        loss.backward()

        optimizer.step()

        #break # debug

    return total_loss / total_step



def eval(args, model, device, loader):
    #return 0
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.subgraph, batch.prompt_mask)

        # """
        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)
        """debug
        x = 0-batch.y.view(pred.shape)[:,:1]
        x[0,0] = -1
        x[1,0] = 1 
        y_true.append(x)
        y_scores.append(pred[:,:2])
        break
        # """

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    return sum(roc_list)/len(roc_list) #y_true.shape[1]



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of fine-tuning of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('-bs', '--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('-e', '--epochs', type=int, default=10,
                        help='number of epochs to train (default: 100 if use linear pred, 10 if hard-coded mapping)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('-prompt', '--graph_prompting', type=str, default="stru",
                        help='graph prompting method (none for nothing, feat for feature, stru for structure, both for both feature and structural)')
    parser.add_argument('-npn', '--num_prompt_nodes', type=int, default=0,
                        help='number of prompt nodes attached to the rest of the graph (default: 0)')
    parser.add_argument('--JK', type=str, default="last", choices=["last", "sum", "max", "concat", "none"],
                        help='how the node features across layers are combined. last, sum, max or concat; none for using another prediction method.')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--dataset', type=str, default = 'tox21', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default = '', help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default = '', help='output filename')
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type = str, default="scaffold", help = "random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', action='store_true', help='evaluating training or not (default: not)')
    parser.add_argument('--num_workers', type=int, default = 4, help='number of workers for dataset loading')
    args = parser.parse_args()


    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    # hyper-parameters for prompting
    feat_prompting = stru_prompting = False
    if args.graph_prompting in ["feat", "both"]:
        feat_prompting = True
    if args.graph_prompting in ["stru", "both"]:
        stru_prompting = True

    #Bunch of classification tasks 
    if args.dataset == "tox21":
        num_tasks = 12
    elif args.dataset == "hiv":
        num_tasks = 1
    #elif args.dataset == "pcba": # can't load, can't run
    #    num_tasks = 128
    elif args.dataset == "muv":
        num_tasks = 17
    elif args.dataset == "bace":
        num_tasks = 1
    elif args.dataset == "bbbp":
        num_tasks = 1
    elif args.dataset == "toxcast":
        num_tasks = 617
    elif args.dataset == "sider":
        num_tasks = 27
    elif args.dataset == "clintox":
        num_tasks = 2
    else:
        raise ValueError("Invalid dataset name.")

    #set up dataset
    dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset)

    print(dataset)

    pre_processor = PreprocessPrompt(dataset, args.num_workers)
    max_nodes = pre_processor.process(num_prompt_nodes=args.num_prompt_nodes)

    #print(dataset.slices["edge_index"])
    #print(dataset.slices["edge_attr"])

    #print(max_nodes)
    #dexit(0)
    
    if args.split == "scaffold":
        smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
        print("scaffold")
    elif args.split == "random":
        train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        print("random")
    elif args.split == "random_scaffold":
        smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        print("random scaffold")
    else:
        raise ValueError("Invalid split option.")

    # print(train_dataset.__dict__) # so far so good, still keeps the modified data
    # exit(0)

    """
    # print(train_dataset[0])
    examine_components = ExamineConnectedComponents()
    tmp_cpn = examine_components(dataset)
    # print(tmp_cpn)
    exit(0)
    """

    #train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    #set up model
    model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type, feat_prompting=feat_prompting, stru_prompting=stru_prompting, max_nodes=max_nodes, num_prompt_nodes=args.num_prompt_nodes)
    if not args.input_model_file == "":
        model.from_pretrained(args.input_model_file, device)
    
    model.to(device)

    #pre_processor.label_mapping(model, device, train_loader) # split the dimensions

    #set up optimizer
    #different learning rate for different part of GNN
    model_param_group = list()
    model_param_group.append({"params": model.gnn.parameters()})
    #model_param_group.append({"params": model.gnn.parameters(), "lr":0.01}) # in structure learning we need higher lr here
    if args.graph_pooling == "attention":
        model_param_group.append({"params": model.pool.parameters(), "lr":args.lr*args.lr_scale})
    if hasattr(model.graph_pred_linear, "parameters"):
        model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    print(optimizer)

    train_acc_list = []
    val_acc_list = []
    test_acc_list = []


    if not args.filename == "":
        fname = 'runs/finetune_cls_runseed' + str(args.runseed) + '/' + args.filename
        #delete the directory if there exists one
        if os.path.exists(fname):
            shutil.rmtree(fname)
            print("removed the existing file.")
        writer = SummaryWriter(fname)

    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        
        train_loss = train(args, model, device, train_loader, optimizer)

        print("====Evaluation")
        msg = "loss: %f " %(train_loss)
        if args.eval_train:
            train_acc = eval(args, model, device, train_loader)
            msg += "train acc: %f " %(train_acc)
        else:
            print("omit the training accuracy computation")
            train_acc = 0
        val_acc = eval(args, model, device, val_loader)
        test_acc = eval(args, model, device, test_loader)
        msg += "val acc: %f test acc: %f" %(val_acc, test_acc)

        #print("loss: %f train: %f val: %f test: %f" %(train_loss, train_acc, val_acc, test_acc))
        print(msg)

        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
        train_acc_list.append(train_acc)

        if not args.filename == "":
            writer.add_scalar('data/train auc', train_acc, epoch)
            writer.add_scalar('data/val auc', val_acc, epoch)
            writer.add_scalar('data/test auc', test_acc, epoch)

        print("")

    if not args.filename == "":
        writer.close()

if __name__ == "__main__":
    main()
