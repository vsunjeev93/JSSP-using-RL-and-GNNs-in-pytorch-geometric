import torch
import torch_geometric.transforms as transform
def next_state(
    data,
    actions,
    param_dict
):
    '''
    machine_last_op- last operation undertaken in a machine -size (batch_num_machines,)
    op_machine- machine used in each operation- size (batch_num_nodes,)
    m_avail_time- machine availability -size (batch_num_machines,)
    op_end_time-operation_end_times- size (batch_num_nodes,)
    processing_times- processing times for each operation size (batch_num_nodes,)
    batch_index_start - for book keeping stores the starting index of each graph size(number of graphs in batch,)
    mask- size(batch_num_nodes,)
    returns data (graph)
    '''

    machine_last_op=data.machine_last_op
    op_machine=data.op_machine_map
    m_avail_time=data.machine_avail_time
    batch_index_start=data.graph_id_offset
    processing_times=data.processing_times
    op_end_time=data.op_end_time
    mask=data.mask
    remaining_processing_time=data.remaining_processing_time
    nm=param_dict['nm']
    nj=param_dict['nj']
    
    # get last operation done on machines used for actions and add the edges
    last_op = machine_last_op[op_machine[actions]]
    non_zero_edges=torch.nonzero(last_op).squeeze(-1)
    if non_zero_edges.numel()!=0:
        edges_to_add = torch.stack((last_op[non_zero_edges], actions[non_zero_edges]))
        data.edge_index = torch.cat([data.edge_index, edges_to_add], dim=1)
        remove_duplicates = transform.RemoveDuplicatedEdges()
        data = remove_duplicates(data)
    # get previous operations in same job to calculate processing times
    prev_actions = actions - 1
    prev_actions = torch.where(
        (prev_actions - batch_index_start) % nm == 0, 0, prev_actions
    )
    # get prev operation in same job's end time
    last_op_end_time = op_end_time[prev_actions]
    # get start times for machines going to be used for actions
    start_times = m_avail_time[op_machine[actions]]
    # get the begin_time of new operations in actions
    new_operation_begin_time, _ = torch.max(
        torch.stack((start_times, last_op_end_time), dim=1), dim=1
    )
    # add processing times to update the operation end times for actions
    op_end_time[actions] = new_operation_begin_time + processing_times[actions]
    # update machine availability times for machines used by current actions
    m_avail_time[op_machine[actions]] = op_end_time[actions]
    # update last operation used by machines- used to add edges in GNN
    machine_last_op[op_machine[actions]] = actions
    # next_possible action
    next_actions = actions + 1
    next_actions = torch.where(
        (next_actions - batch_index_start) % nm == 1, float("nan"), next_actions
    )
    next_actions = next_actions[~torch.isnan(next_actions)].int()
    mask[actions.squeeze()] = False
    mask[next_actions.squeeze()] = True
    candidate_actions=torch.nonzero(mask)
    previous_candidate_actions=candidate_actions-1
    previous_candidate_actions=torch.where(
        (previous_candidate_actions) % (nm*nj+2) == 0, 0, previous_candidate_actions
    ) 
    # return the updated tensors
    # est end time update
    new_est=op_end_time[actions]+remaining_processing_time[actions]
    stacked=torch.stack((data.est_end_time,new_est),dim=1)
    data.est_end_time,_=torch.max(stacked,dim=1)
    data.machine_last_op=machine_last_op
    data.op_machine_map=op_machine
    data.mach_avail_time=m_avail_time
    data.op_end_time=op_end_time
    data.mask=mask
    return data