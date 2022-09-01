import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros

num_atom_type = 120 #including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6 #including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3 

class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not. 
        

    See https://arxiv.org/abs/1810.00826
    """
    def __init__(self, emb_dim, aggr = "add"):
        super(GINConv, self).__init__()
        #multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GCNConv(MessagePassing):

    def __init__(self, emb_dim, aggr = "add"):
        super(GCNConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        norm = self.norm(edge_index, x.size(0), x.dtype)

        x = self.linear(x)

        return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings, norm = norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)


class GATConv(MessagePassing):
    def __init__(self, emb_dim, heads=2, negative_slope=0.2, aggr = "add"):
        super(GATConv, self).__init__()

        self.aggr = aggr

        self.emb_dim = emb_dim
        self.heads = heads
        self.negative_slope = negative_slope

        self.weight_linear = torch.nn.Linear(emb_dim, heads * emb_dim)
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))

        self.bias = torch.nn.Parameter(torch.Tensor(emb_dim))

        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, heads * emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, heads * emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):

        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        x = self.weight_linear(x).view(-1, self.heads, self.emb_dim)
        return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, edge_index, x_i, x_j, edge_attr):
        edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        x_j += edge_attr

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        aggr_out = aggr_out.mean(dim=1)
        aggr_out = aggr_out + self.bias

        return aggr_out


class GraphSAGEConv(MessagePassing):
    def __init__(self, emb_dim, aggr = "mean"):
        super(GraphSAGEConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        x = self.linear(x)

        return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p = 2, dim = -1)



class GNN(torch.nn.Module):
    """
    

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """
    def __init__(self, num_layer, emb_dim, JK = "last", drop_ratio = 0, gnn_type = "gin", feat_prompting=False, stru_prompting=False, max_nodes=0, num_prompt_nodes=0):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        self.feat_prompting = feat_prompting
        self.stru_prompting = stru_prompting
        self.num_prompt_nodes = num_prompt_nodes
        self.mlp_virtualnode_list = None
        self.mlp_softmax_list = None
        if self.feat_prompting and not self.stru_prompting:
            # print(num_atom_type, num_chirality_tag) # 120, 3
            self.max_nodes = max_nodes
            self.prompt_embed = torch.nn.Embedding(self.max_nodes, emb_dim)
            #self.x_embedding1.requires_grad = False
            #self.x_embedding2.requires_grad = False
            #torch.nn.init.xavier_uniform_(self.prompt_embed.weight.data)
            torch.nn.init.zeros_(self.prompt_embed.weight.data)
        elif self.stru_prompting and not self.feat_prompting:
            self.prompt_embed = torch.nn.Embedding(self.num_prompt_nodes, emb_dim) # single virtual node
            torch.nn.init.xavier_uniform_(self.prompt_embed.weight.data)
            
            """
            ### List of MLPs to transform virtual node at every layer
            self.mlp_virtualnode_list = torch.nn.ModuleList()
            #for layer in range(self.num_prompt_nodes):#range(num_layer - 1): # prompt specific
            for layer in range(num_layer - 1): # layer-specific
                self.mlp_virtualnode_list.append(torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), \
                                                    torch.nn.Linear(2*emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU()))

            #self.mlp_softmax_list = torch.nn.ModuleList()
            #for layer in range(self.num_prompt_nodes):#range(num_layer - 1): # prompt specific
            #    self.mlp_softmax_list.append(torch.nn.Sequential(torch.nn.Linear(emb_dim, 1), torch.nn.BatchNorm1d(1), torch.nn.ReLU()))

            """

        #elif self.stru_prompting and self.feat_prompting:
        #    assert False, "not implemented"


        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr = "add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    #def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 6:
            x, edge_index, edge_attr, batch, subgraph, promt_mask = argv[0], argv[1], argv[2], argv[3], argv[4], argv[5]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch, subgraph, promt_mask = data.x, data.edge_index, data.edge_attr, data.batch, data.subgraph, data.prompt_mask
        else:
            raise ValueError("unmatched number of arguments.")
        #print(subgraph)
        #print(self.prompt_embed.weight)
        #input()
        #exit(0)

        #print(x[:,0][promt_mask])
        #print(promt_mask)

        prompt_index = x[:,0]

        x = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1]) # the combination of two embedding parts makes the final embedding
        if self.feat_prompting and not self.stru_prompting:
            #x += self.prompt_embed(torch.remainder(x[:,0], self.max_nodes).long() )
            x += self.prompt_embed(torch.remainder(subgraph, self.max_nodes).long() )
        elif self.num_prompt_nodes > 0:
            ### virtual node embeddings for graphs
            #virtualnode_embedding = self.prompt_embed(torch.zeros(x.shape[0]).to(edge_index.dtype).to(x.device))
            # virtualnode_embedding = self.prompt_embed(torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))
            
            #print(prompt_index[promt_mask])
            #print(x[promt_mask].shape, promt_mask[promt_mask==True].shape)
            x[promt_mask] = self.prompt_embed(prompt_index[promt_mask])
            #print("next challenge: merge the two embeddings together")
            #exit(0)

            """
            virtualnode_embedding = self.prompt_embed(torch.arange(self.num_prompt_nodes).repeat(batch[-1].item() + 1, 1).to(edge_index.dtype).to(edge_index.device))
            all_embeddings = list()
            for i in range(self.num_prompt_nodes):
                all_embeddings.append(virtualnode_embedding[:,i,:])
            #virtualnode_embedding = virtualnode_embedding[:,0,:]
            #print(torch.arange(self.num_prompt_nodes).repeat(batch[-1].item() + 1, 1).shape)
            #exit(0)
            #print(virtualnode_embedding.shape)
            #exit(0)
            """

        """
        h_list = [x]
        for layer in range(self.num_layer):

            h = self.gnns[layer](h_list[layer], edge_index, edge_attr) 
            for i in range(self.num_prompt_nodes):
                h = h + all_embeddings[i][batch] / self.num_prompt_nodes
            h = self.batch_norms[layer](h)
            #h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)

            ### update the virtual nodes
            if layer < self.num_layer - 1:
                for i in range(self.num_prompt_nodes):
                    ### add message from graph nodes to virtual nodes
                    s = 0.5
                    
                    # all_embeddings[i] = (1-s) * global_mean_pool(h, batch) + s * all_embeddings[i]

                    #score = self.mlp_softmax_list[layer](all_embeddings[i])
                    #print(score)
                    #exit(0)

                    all_embeddings[i] = (1-s) * global_mean_pool(h, batch) + s * all_embeddings[i]

                    # sample code of applying softmax
                    #from torch_geometric.utils import softmax
                    #score = softmax(h[:,0], batch)
                    #print(score)
                    #print(batch)
                    #print(global_add_pool(score, batch)) # correct
                    #exit(0)

                    ### transform virtual nodes using MLP
                    # residual? virtualnode_embedding + 
                    all_embeddings[i] = F.dropout(self.mlp_virtualnode_list[layer](all_embeddings[i]), self.drop_ratio, training = self.training)
        #"""


        #"""
        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr) #+ virtualnode_embedding
            h = self.batch_norms[layer](h)
            #h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)
        #"""
        

        """
        h_list = [x]
        for layer in range(self.num_layer):

            h = self.gnns[layer](h_list[layer], edge_index, edge_attr) 
            if self.num_prompt_nodes > 0:
                h = h + virtualnode_embedding[batch]
            h = self.batch_norms[layer](h)
            #h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)

            ### update the virtual nodes
            if self.num_prompt_nodes > 0 and layer < self.num_layer - 1:
                ### add message from graph nodes to virtual nodes
                s = 0.5
                virtualnode_embedding = (1-s) * global_mean_pool(h, batch) + s * virtualnode_embedding
                ### transform virtual nodes using MLP
                # residual? virtualnode_embedding + 
                virtualnode_embedding = F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding), self.drop_ratio, training = self.training)
        #"""

                
        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == "none":
            node_representation = h_list[-1]

        # print(node_representation.shape) # torch.Size([513, 300])
        return node_representation


class GNN_graphpred(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type: gin, gcn, graphsage, gat
        
    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """
    def __init__(self, num_layer, emb_dim, num_tasks, JK = "last", drop_ratio = 0, graph_pooling = "mean", gnn_type = "gin", feat_prompting=False, stru_prompting=True, max_nodes=0, num_prompt_nodes=0):
        super(GNN_graphpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.feat_prompting = feat_prompting #and not stru_prompting
        self.stru_prompting = stru_prompting #and not feat_prompting
        self.freezeGnnParam = feat_prompting and stru_prompting
        self.finetuneGnnMod = not feat_prompting and not stru_prompting
        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type = gnn_type, feat_prompting=feat_prompting, stru_prompting=stru_prompting, max_nodes=max_nodes, num_prompt_nodes=num_prompt_nodes)

        for param in self.gnn.parameters():
            param.requires_grad = False

        #Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        #For graph-level binary classification
        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1
        
        if self.JK == "concat":
            self.output_dim = self.mult * (self.num_layer + 1) * self.emb_dim
            self.graph_pred_linear = torch.nn.Linear(self.output_dim, self.num_tasks)
        elif self.JK == "none": # TODO: to improve this # TODO: randomly k? 
            #self.indices = list(range(self.num_tasks)) #torch.LongTensor(list(range(self.num_tasks)))
            self.output_dim = self.mult * self.emb_dim
            self.max_k = int(self.output_dim / self.num_tasks) # floor
            self.indices = list(range(self.num_tasks * self.max_k))
            self.graph_pred_linear = self.graph_pred_hard_coded 
        else:
            self.output_dim = self.mult * self.emb_dim
            self.graph_pred_linear = torch.nn.Linear(self.output_dim, self.num_tasks)
        
        # apply feature prompting
        if self.feat_prompting and not self.stru_prompting:
            # prompt_embed is the only thing to update (instead of fine-tuning)
            for param in self.gnn.parameters(): # self.parameters(): 
                param.requires_grad = False
            self.gnn.prompt_embed.weight.requires_grad = True 
        elif self.stru_prompting and not self.feat_prompting:
            # prompt_embed is the only thing to update (instead of fine-tuning)
            for param in self.gnn.parameters():
                param.requires_grad = False
            self.gnn.prompt_embed.weight.requires_grad = True
            if self.gnn.num_prompt_nodes > 0 and self.gnn.mlp_virtualnode_list:
                for name, param in self.gnn.named_parameters():
                    if "mlp_virtualnode_list" in name:
                        param.requires_grad = True
                if self.gnn.mlp_softmax_list:
                    for name, param in self.gnn.named_parameters():
                        if "mlp_softmax_list" in name:
                            param.requires_grad = True

        elif not self.feat_prompting and not self.stru_prompting:
            for param in self.gnn.parameters(): # self.parameters():
                param.requires_grad = True # when testing the frozen mode, set it to False
        else:
            for param in self.gnn.parameters(): # self.parameters():
                param.requires_grad = False #True # when testing the frozen mode, set it to False
            """
            for name, param in self.gnn.named_parameters():
                if "gnns.4" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            """

    def graph_pred_hard_coded(self, output):
        #print(output.requires_grad)   
        output = torch.mean(torch.reshape(output[:,self.indices], (output[:,self.indices].shape[0], self.num_tasks, -1)), dim=2)
        #print(output.requires_grad, output.shape)
        #exit(0)
        return output

    def from_pretrained(self, model_file, device):
        #self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        self.gnn.load_state_dict(torch.load(model_file, map_location=device), strict=False) # not strict, then we can add prompt

        """
        # apply feature prompting
        if self.feat_prompting:
            # prompt_embed is the only thing to update (instead of fine-tuning)
            for param in self.parameters(): # self.gnn.parameters():
                param.requires_grad = False
            self.gnn.prompt_embed.weight.requires_grad = True 
        else: 
            # PX: the code they gave us were somewhat buggy here, GNN's parameters' require_grad were all false
            for param in self.gnn.parameters():
                param.requires_grad = True
        """

        
        debug_print = False # True
        if debug_print:
            # debug
            for name, param in self.gnn.named_parameters(): # self.named_parameters():
                if param.requires_grad:
                    print("requires grad", name, param.data.shape)
                else:
                    print("no grad", name) # param.data
            exit(0)
        

    def forward(self, *argv):

        # print(self.gnn.prompt_embed(torch.LongTensor([0.])))
        #print(self.gnn.x_embedding1(torch.LongTensor([0.])))
        #input()
        if len(argv) == 6:
            x, edge_index, edge_attr, batch, subgraph, promt_mask = argv[0], argv[1], argv[2], argv[3], argv[4], argv[5]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch, subgraph, promt_mask = data.x, data.edge_index, data.edge_attr, data.batch, data.subgraph, data.prompt_mask
        else:
            raise ValueError("unmatched number of arguments.")
        # print(x.shape) # torch.Size([513, 2])
        # exit(0)

        node_representation = self.gnn(x, edge_index, edge_attr, batch, subgraph, promt_mask)

        # print(node_representation.shape) # torch.Size([513, 300])

        # print(self.pool(node_representation, batch).shape) # torch.Size([batch_size, emb_dim])
        # print(self.graph_pred_linear(self.pool(node_representation, batch)).shape) # torch.Size([batch_size, num_task])
        # exit(0)

        #print(node_representation.requires_grad)

        return self.graph_pred_linear(self.pool(node_representation, batch))


    def representation(self, *argv):
        if len(argv) == 5:
            x, edge_index, edge_attr, batch, subgraph = argv[0], argv[1], argv[2], argv[3], argv[4]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch, subgraph = data.x, data.edge_index, data.edge_attr, data.batch, data.subgraph
        else:
            raise ValueError("unmatched number of arguments.")
        node_representation = self.gnn(x, edge_index, edge_attr, batch, subgraph)

        return self.pool(node_representation, batch)


if __name__ == "__main__":
    pass

