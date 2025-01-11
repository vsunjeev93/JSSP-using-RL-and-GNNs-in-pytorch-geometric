import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import index_to_mask


def permute_rows(x):
    """
    x is a np array
    """
    ix_i = np.tile(np.arange(x.shape[0]), (x.shape[1], 1)).T
    ix_j = np.random.sample(x.shape).argsort(axis=1)
    return x[ix_i, ix_j]


def uni_instance_gen(nj, nm, low, high):
    times = np.random.randint(low=low, high=high, size=(nj, nm))
    machines = np.expand_dims(np.arange(0, nm), axis=0).repeat(repeats=nj, axis=0)
    machines = permute_rows(machines)
    return times, machines


class JSSPData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if "index" in key:
            return self.num_nodes
        elif "op_machine_map" in key:
            return self.num_mach
        elif "machine_last_op" in key:
            return self.num_nodes
        elif "graph_id_offset" in key:
            return self.num_nodes
        else:
            return 0


def __cat_dim__(self, key, value, *args, **kwargs):
    if "index" in key or key in [
        "op_machine_map",
        "machine_last_op",
        "op_end_time",
        "graph_id_offset",
        "machine_avail_time",
        "mask",
        "est_end_time",
    ]:
        return 1
    else:
        return 0


def generate_graph_from_data(nj, nm, times, machines):

    edges = []
    nodes = 1
    features = [torch.tensor([0] * (nm + 3))]
    op_machine = [-1] * (nm * nj + 2)
    processing_times = [0] * (nm * nj + 2)
    candidate_actions = []
    remaining_processing_time = [0] * (nm * nj + 2)
    est_end_time = np.max(np.cumsum(times, axis=1))
    reverse_edges = []
    for i in range(nj):
        edges.append(torch.tensor([0, nodes]))
        reverse_edges.append(torch.tensor([nodes, 0]))
        candidate_actions.append(nodes)
        for j in range(nm):
            machine_vector = [0] * nm
            machine_vector[machines[i, j]] = 1
            op_machine[nodes] = int(machines[i, j])
            processing_times[nodes] = times[i, j]
            if j != nm - 1:
                remaining_processing_time[nodes] = np.sum(times[i, j + 1 :])
            features.append(
                torch.tensor(
                    [times[i, j], remaining_processing_time[nodes], nm - j - 1]
                    + machine_vector
                )
            )
            if (nodes) % nm != 0:
                edges.append(torch.tensor([nodes, nodes + 1]))
                reverse_edges.append(torch.tensor([nodes + 1, nodes]))
            else:
                edges.append(torch.tensor([nodes, nj * nm + 1]))
                reverse_edges.append(torch.tensor([nj * nm + 1, nodes]))
            nodes += 1
    features.append(torch.tensor([0] * (nm + 3)))
    nodes += 1
    mask = index_to_mask(torch.tensor(candidate_actions), size=nodes)
    edge_index = torch.stack(edges)
    reverse_edge_index = torch.stack(reverse_edges)
    features = torch.stack(features)
    graph = JSSPData(
        x=features,
        edge_index=edge_index.t().contiguous(),
        reversed_edge_index=reverse_edge_index.t().contiguous(),
        op_machine_map=torch.tensor(op_machine).int(),
        processing_times=torch.tensor(processing_times),
        num_mach=nm,
        machine_avail_time=torch.tensor([0] * nm),
        machine_last_op=torch.tensor([0] * nm),
        op_end_time=torch.tensor([0] * nodes),
        graph_id_offset=torch.tensor([0]),
        mask=mask,
        remaining_processing_time=torch.tensor(remaining_processing_time).float(),
        est_end_time=torch.tensor(est_end_time).float(),
    )
    return graph


def data_generator(nj, nm, low, high, instances=10000, batch_size=12):
    graphs = []
    np.random.seed(42)
    for instance in range(instances):
        times, machines = uni_instance_gen(nj, nm, low, high)
        graph = generate_graph_from_data(nj, nm, times, machines)
        graphs.append(graph)
    loader = DataLoader(graphs, batch_size=batch_size)
    return loader


loader = data_generator(5, 5, 1, 3, 10, 2)
