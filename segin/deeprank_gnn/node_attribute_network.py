from torch_geometric.nn import MessagePassing
import torch

class NodeAttributeNetwork(MessagePassing):
    """
        Computes the node and edge attributes based on relative positions
    """

    def __init__(self):
        super(NodeAttributeNetwork, self).__init__(node_dim=-2, aggr="add")  # <---- Mean of all edge features

    def forward(self,x,  edge_index, edge_attr):
        """ Simply sums the edge attributes """
        num_nodes = x.size(0)
        node_attr = torch.zeros(num_nodes, edge_attr.size(1)).to(x.device)
        node_attr_values = self.propagate(edge_index,node_attr=node_attr, edge_attr=edge_attr)  # TODO: continue here!
        node_attr[0:node_attr_values.shape[0],:] = node_attr_values 
        return node_attr

    def message(self, edge_attr):
        """ The message is the edge attribute """
        return edge_attr

    def update(self, node_attr):
        """ The input to update is the aggregated messages, and thus the node attribute """
        return node_attr
