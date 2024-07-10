import torch
from torch.nn import Parameter
import torch.nn.functional as F
import torch.nn as nn

#from torch_scatter import scatter_mean
#from torch_scatter import scatter_sum

from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

# torch_geometric import
from torch_geometric.nn.inits import uniform
from torch_geometric.nn import max_pool_x, MessagePassing
from torch_geometric.data import DataLoader

# deeprank_gnn import
from deeprank_gnn.community_pooling import get_preloaded_cluster, community_pooling
from deeprank_gnn.DataSet import HDF5DataSet, PreCluster

from e3nn.o3 import Irreps, spherical_harmonics
from .o3_building_blocks import O3TensorProduct, O3TensorProductSwishGate
from .node_attribute_network import NodeAttributeNetwork
from deeprank_gnn.balanced_irreps import WeightBalancedIrreps
from e3nn.nn import BatchNorm
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool

class SEGINet(torch.nn.Module):
    # input_shape -> number of node input features
    # output_shape -> number of output value per graph
    # input_shape_edge -> number of edge input features
    def __init__(self, node_feature,
                 hidden_feature=128,
                 output_feature=1,
                 N=3, 
                 lmax_pos = 3):
        super().__init__()
        node_in_irreps = Irreps(f"{node_feature}x0e")
        node_hidden_irreps_scalar = Irreps(f"{hidden_feature}x0e")
        attr_irreps = Irreps.spherical_harmonics(lmax_pos)
        node_out_irreps_scalar = Irreps(f"{output_feature}x0e") 
        #print(f"node_out_irreps_scalar: {node_out_irreps_scalar}")       
        node_hidden_irreps = WeightBalancedIrreps(
            node_hidden_irreps_scalar, attr_irreps, True, lmax=lmax_pos)
        #MessagePassing for node feature
        self.node_attribute_net = NodeAttributeNetwork()
        self.attr_irreps = Irreps.spherical_harmonics(lmax_pos)

        self.node_embedding_in = O3TensorProductSwishGate(node_in_irreps,  # in
                                                          node_hidden_irreps, # out
                                                          self.attr_irreps)
        self.node_embedding_out = O3TensorProduct(node_hidden_irreps,  # in
                                                 node_hidden_irreps,  # out
                                                 self.attr_irreps)         # steerable attribute
        self.in_layers = []
        for i in range(N):
            self.in_layers.append(SEGINLayer(node_hidden_irreps,  # in
                                     node_hidden_irreps,  # hidden
                                     node_hidden_irreps,  # out
                                     self.attr_irreps,         # steerable attribute
                                     ))  
        self.in_layers = nn.ModuleList(self.in_layers)  

        self.out_layers = [] 
        for i in range(N):
            self.out_layers.append(SEGINLayer(node_hidden_irreps,  # in
                                     node_hidden_irreps,  # hidden
                                     node_hidden_irreps,  # out
                                     self.attr_irreps,         # steerable attribute
                                     ) )   
        self.out_layers = nn.ModuleList(self.out_layers)

        self.pre_pool_layer = O3TensorProductSwishGate(node_hidden_irreps,           # in
                                                              node_hidden_irreps_scalar,    # out
                                                              attr_irreps)
        self.post_pool_layer_1 = O3TensorProductSwishGate(node_hidden_irreps_scalar,
                                                               node_hidden_irreps_scalar)
        self.post_pool_layer_2 = O3TensorProduct(node_hidden_irreps_scalar,
                                                      node_out_irreps_scalar) 
        self.feature_norm = BatchNorm(node_out_irreps_scalar)

    def forward(self, data):

        x, pos,batch = data.x, data.pos, data.batch

        #internal attributes
        in_edge_attr_dist = data.internal_edge_attr #it is dist by default
        in_edge_index = data.internal_edge_index
        #relative displacement
        in_rel_pos = pos[in_edge_index[0]] - pos[in_edge_index[1]]
        in_edge_attr = spherical_harmonics(self.attr_irreps, in_rel_pos, normalize=True, normalization='component')

        in_node_feat = self.node_attribute_net(x, in_edge_index, in_edge_attr)
        x = self.node_embedding_in(x, in_node_feat)
        for layer in self.in_layers:
            x, pos = layer(x, pos, in_edge_index, in_edge_attr_dist, in_edge_attr, in_node_feat, batch)

        #external attributes
        ex_edge_attr_dist = data.edge_attr
        ex_edge_index = data.edge_index
        #relative displacement
        ex_rel_pos = pos[ex_edge_index[0]] - pos[ex_edge_index[1]]
        ex_edge_attr = spherical_harmonics(self.attr_irreps, ex_rel_pos, normalize=True, normalization='component')
        ex_node_feat = self.node_attribute_net(x, ex_edge_index, ex_edge_attr)  
        x = self.node_embedding_out(x, ex_node_feat)  
        for layer in self.out_layers:
            x, pos = layer(x, pos, ex_edge_index, ex_edge_attr_dist, ex_edge_attr, ex_node_feat, batch)
        x = self.pre_pool_layer(x, ex_node_feat)
        x = global_add_pool(x,data.batch)
        x = self.post_pool_layer_1(x)
        #print(x.shape)
        #x = self.dropout(x)
        x = self.post_pool_layer_2(x)
        #print(x.shape)
        #x = self.feature_norm(x)
        #x = torch.squeeze(self.out(x))
        return x
    

class SEGINLayer(MessagePassing):
    def __init__(self,node_in_irreps,node_hidden_irreps,node_out_irreps,attr_irreps):
        super(SEGINLayer, self).__init__(node_dim=-2,aggr= "add")
        irreps_input = (node_in_irreps+ node_in_irreps + Irreps("1x0e")).simplify()
        self.message_layer_1 = O3TensorProductSwishGate(irreps_input,
                                                        node_hidden_irreps,
                                                        attr_irreps)
        self.message_layer_2 = O3TensorProductSwishGate(node_hidden_irreps,
                                                        node_hidden_irreps,
                                                        attr_irreps)
        irreps_update_in = (node_in_irreps + node_hidden_irreps).simplify()
        self.update_layer_1 = O3TensorProductSwishGate(irreps_update_in,
                                                       node_out_irreps,
                                                       attr_irreps)
        self.inference_layer = O3TensorProduct(node_hidden_irreps, Irreps("1x0e"), attr_irreps)
        self.feature_norm = BatchNorm(node_hidden_irreps)

    def forward(self,x,pos,edge_index, edge_dist, edge_attr, node_attr, batch):
        x, pos = self.propagate(edge_index, x=x, pos=pos, edge_dist=edge_dist,
                                node_attr=node_attr, edge_attr=edge_attr)
        return x,pos

    def message(self,x_i,x_j, edge_dist, edge_attr):
        message = self.message_layer_1(torch.cat((x_i, x_j, edge_dist), dim= -1), edge_attr)
        message = self.message_layer_2(message, edge_attr)
        message = self.feature_norm(message)
        attention = torch.sigmoid(self.inference_layer(message,edge_attr))
        message = message*attention
        return message
    
    def update(self, message, x, pos, node_attr):
        att = F.softmax(message, dim=1 )
        x = att * x
        update = self.update_layer_1(torch.cat((x,message), dim= -1), node_attr)
        x = update 
        return x, pos
