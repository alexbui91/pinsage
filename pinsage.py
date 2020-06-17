import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from movielens import MovieLens


def get_embeddings(h, nodeset):
    return h[nodeset]

def put_embeddings(h, nodeset, new_embeddings):
    n_nodes = nodeset.shape[0]
    n_features = h.shape[1]
    return h.scatter(0, nodeset[:, None].expand(n_nodes, n_features), new_embeddings)

def random_walk_sampler(G, nodeset, n_traces, n_hops):
    '''
    G: DGLGraph
    nodeset: 1D CPU Tensor of node IDs
    n_traces: int
    n_hops: int
    return: 3D CPU Tensor or node IDs (n_nodes, n_traces, n_hops + 1)
    '''
    n_nodes = nodeset.shape[0]
    traces = torch.zeros(n_nodes, n_traces, n_hops + 1, dtype=torch.int64)

    for i in range(n_nodes):
        for j in range(n_traces):
            cur = nodeset[i]
            for k in range(n_hops + 1):
                traces[i, j, k] = cur
                neighbors = G.successors(cur)
                assert neighbors.shape[0] > 0
                cur = neighbors[torch.randint(len(neighbors), ())]

    return traces

def random_walk_distribution(G, nodeset, n_traces, n_hops):
    n_nodes = nodeset.shape[0]
    n_available_nodes = G.number_of_nodes()
    traces = random_walk_sampler(G, nodeset, n_traces, n_hops)
    visited_nodes = traces[:, :, 1:].view(n_nodes, -1)  # (n_nodes, n_visited_other_nodes)
    visited_counts = (
            torch.zeros(n_nodes, n_available_nodes)
            .scatter_add_(1, visited_nodes, torch.ones_like(visited_nodes, dtype=torch.float64)))
    visited_prob = visited_counts / visited_counts.sum(1, keepdim=True)
    return visited_prob

def random_walk_distribution_topt(G, nodeset, n_traces, n_hops, top_T):
    '''
    returns the top T important neighbors of each node in nodeset, as well as
    the weights of the neighbors.
    '''
    visited_prob = random_walk_distribution(G, nodeset, n_traces, n_hops)
    return visited_prob.topk(1, top_T)

def random_walk_nodeflow(G, nodeset, n_layers, n_traces, n_hops, top_T):
    '''
    returns a list of triplets (
        "active" node IDs whose embeddings are computed at the i-th layer (num_nodes,)
        weight of each neighboring node of each "active" node on the i-th layer (num_nodes, top_T)
        neighboring node IDs for each "active" node on the i-th layer (num_nodes, top_T)
    )
    '''
    nodeflow = []
    cur_nodeset = nodeset
    for i in reversed(range(n_layers)):
        nb_weights, nb_nodes = random_walk_distribution_topt(G, nodeset, n_traces, n_hops, top_T)
        nodeflow.insert((cur_nodeset, nb_weights, nb_nodes))
        cur_nodeset = torch.cat([nb_nodes.view(-1), cur_nodeset]).unique()

    return nodeflow

class PinSageConv(nn.Module):
    def __init__(self, in_features, out_features, hidden_features):
        super(PinSageConv, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features

        self.Q = nn.Linear(in_features, hidden_features)
        self.W = nn.Linear(in_features + hidden_features, out_features)

    def forward(self, h, nodeset, nb_nodes, nb_weights):
        '''
        h: node embeddings (num_total_nodes, in_features), or a container
           of the node embeddings (for distributed computing)
        nodeset: node IDs in this minibatch (num_nodes,)
        nb_nodes: neighbor node IDs of each node in nodeset (num_nodes, num_neighbors)
        nb_weights: weight of each neighbor node (num_nodes, num_neighbors)
        return: new node embeddings (num_nodes, out_features)
        '''
        n_nodes, T = nb_nodes.shape[0]

        h_nodeset = get_embeddings(h, nodeset)  # (n_nodes, in_features)
        h_neighbors = get_embeddings(h, nb_nodes.view(-1)).view(n_nodes, T, self.in_features)

        h_neighbors = F.relu(self.Q(h_neighbors))
        h_agg = (nb_weights[:, :, None] * h_neighbors).sum(1) / nb_weights.sum(1, keepdim=True)

        h_concat = torch.cat([h_nodeset, h_agg], 1)
        h_new = F.relu(self.W(h_concat))
        h_new /= h_new.norm(dim=1, keepdim=True)

        return h_new

class PinSage(nn.Module):
    '''
    Completes a multi-layer PinSage convolution
    G: DGLGraph
    feature_sizes: the dimensionality of input/hidden/output features
    T: number of neighbors we pick for each node
    n_traces: number of random walk traces to generate during sampling
    n_hops: number of hops of each random walk trace during sampling
    '''
    def __init__(self, G, feature_sizes, T, n_traces, n_hops):
        super(PinSage, self).__init__()

        self.G = G
        self.T = T
        self.n_traces = n_traces
        self.n_hops = n_hops

        self.in_features = feature_sizes[0]
        self.out_features = feature_sizes[-1]
        self.n_layers = len(feature_sizes) - 1

        self.convs = nn.ModuleList()
        for i in range(self.n_layers):
            self.convs.append(PinSageConv(
                feature_sizes[i], feature_sizes[i+1], feature_sizes[i+1]))

    def forward(self, h, nodeset):
        '''
        Given a complete embedding matrix h and a list of node IDs, return
        the output embeddings of these node IDs.

        h: node embeddings (num_total_nodes, in_features), or a container
           of the node embeddings (for distributed computing)
        nodeset: node IDs in this minibatch (num_nodes,)
        return: new node embeddings (num_nodes, out_features)
        '''
        nodeflow = random_walk_nodeflow(self.G, nodeset, self.n_layers, self.n_traces, self.n_hops, self.T)
        
        for i, (nodeset, nb_weights, nb_nodes) in enumerate(nodeflow):
            new_embeddings = self.convs[i](h, nodeset, nb_nodes, nb_weights)
            h = put_embeddings(h, nodeset, new_embeddings)

        return get_embeddings(h, nodeset)


if __name__ == "__main__":
    directory = "data/ml-1m"
    m = MovieLens(directory)
    ps = PinSage(m.todglgraph(), [128, 128, 1], 10, 25, 10)