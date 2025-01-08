from data_generator import data_generator
from actor import actor
from critic import critic
import torch
import torch_geometric.transforms as transform
from torch_geometric.utils import mask_to_index
def next_state(
    data,
    actions,
    param_dict
):
    # machine_last_op- last operation undertaken in a machine -size (batch_nm,)
    # op_machine- machine used in each operation- size (batch_num_nodes,)
    # m_avail_time- machine availability -size (batch_nm,)
    # op_end_time-operation_end_times- size(batch_num_nodes,)
    # processing_times- size (batch_num_nodes,)
    # batch_index_start - size(number of graphs in batch,)
    # mask- size(batch_num_nodes,)
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
        # print(last_op[non_zero_edges],actions[non_zero_edges],edges_to_add,last_op)
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
    # print(mask.size(),next_actions.size())
    mask[next_actions.squeeze()] = True
    candidate_actions=torch.nonzero(mask)
    previous_candidate_actions=candidate_actions-1
    
    previous_candidate_actions=torch.where(
        (previous_candidate_actions) % (nm*nj+2) == 0, 0, previous_candidate_actions
    ) 
    # print(previous_candidate_actions.size(),batch_index_start.size())
    candidate_actions_ready_time=op_end_time[previous_candidate_actions]
    machine_ready_time=m_avail_time[candidate_actions]
    waiting_times=machine_ready_time-candidate_actions_ready_time
    waiting=waiting_times>0
    # print(waiting.size(),candidate_actions.size(),candidate_actions.size(),machine_ready_time.size())
    waiting_candidates=candidate_actions[waiting]
    # print(data.x[:,3])
    # data.x[waiting_candidates,3]=waiting_times[waiting]
    # print(data.x[:,3])



    # print(mask)
    # return the updated tensors

    # est end time update: calculate new for jobs with ope
    # rations seelcted (in action) and compare it with existing estimates(choose max)
    # print(remaining_processing_time,data.est_end_time,actions,op_end_time,processing_times)
    new_est=op_end_time[actions]+remaining_processing_time[actions]
    # print(new_est,data.est_end_time,torch.stack((new_est,data.est_end_time),dim=1))
    stacked=torch.stack((data.est_end_time,new_est),dim=1)
    data.est_end_time,_=torch.max(stacked,dim=1)
    # print(data.est_end_time)
    # assert 1==2
    data.machine_last_op=machine_last_op
    data.op_machine_map=op_machine
    data.mach_avail_time=m_avail_time
    data.op_end_time=op_end_time
    data.mask=mask
    return data
param_dict={'nj':10,
            'nm':10,
            'low':1,
            'high':99,
            'instances':300,
            'batch_size':10,}
device=torch.device('mps')
input_features=param_dict['nm']+3
actor=actor(input_features,512)
actor.to(device)

critic=critic(input_features,512)
critic.to(device)
torch.manual_seed(42)
op_order=[]
nodes_per_graph=param_dict['nj']*param_dict['nm']
LR=0.0001
actor_optim=torch.optim.Adam(actor.parameters(),lr=LR)
critic_optim=torch.optim.Adam(critic.parameters(),lr=LR)
scheduler_actor=torch.optim.lr_scheduler.StepLR(actor_optim,step_size=20,gamma=0.9)
scheduler_critic=torch.optim.lr_scheduler.StepLR(critic_optim,step_size=20,gamma=0.9)
epoch=200
num_iterations=0
batch_num=00
def train_episode(rewards,value_functions,log_actions):
    R=0
    gamma=1
    returns=[]
    actor_loss=[]
    critic_loss=[]
    for r in rewards[::-1]:
        R=r+R*gamma
        returns.insert(0,R)
    for (R,value,log_prob) in zip(returns,value_functions,log_actions):
        advantage=R-value
        actor_loss.append(advantage.detach()*log_prob*-1)
        critic_loss.append(torch.nn.functional.mse_loss(R,value))
        # actor_loss=-advantage*torch.tensor(log_actions)
        # critic_loss=torch.nn.functional.mse_loss(R,value)
    actor_loss=torch.stack(actor_loss)
    critic_loss=torch.stack(critic_loss)
    return actor_loss.sum().mean(),critic_loss.sum().mean()






for e in range(epoch):
    loader = data_generator(*param_dict.values())
    total_reward_epoch=[]
    make_spans=[]
    for data in loader:
        data=data.to(device)
        batch_num+=1
        total_reward=0
        if input_features!=data.x.size(1):
            print(input_features,data.x.size(1))
            raise Exception
        rewards=[]
        value_functions=[]
        log_actions_list=[]
        for i in range(nodes_per_graph):
            actions,log_actions=actor(data)
            value=critic(data)
            # print(
            #     '\nop_machine_map\n',data.op_machine_map,
            #     '\nest_end_time\n',data.est_end_time,
            #     '\nmachine_avail_time\n',data.machine_avail_time,
            #     '\nmachine_last_op\n',data.machine_last_op,
            #     '\ngraph_id_offset\n',data.graph_id_offset,
            #     '\nnum_nodes\n',data.num_nodes,
            #     '\nprocessing_times\n',data.processing_times,
            #     '\nmask\n',data.mask
            # )
            est_makespan=data.est_end_time
            data=next_state(data,actions,param_dict)
            # print('\nactions\n',actions)
            # print('AFTER',
            #     '\nop_machine_map\n',data.op_machine_map,
            #     '\nest_end_time\n',data.est_end_time,
            #     '\nmachine_avail_time\n',data.machine_avail_time,
            #     '\nmachine_last_op\n',data.machine_last_op,
            #     '\ngraph_id_offset\n',data.graph_id_offset,
            #     '\nnum_nodes\n',data.num_nodes,
            #     '\nprocessing_times\n',data.processing_times,
            #     '\nmask\n',data.mask
            # )

            # if i==1:
            #     break
            reward=est_makespan-data.est_end_time
            rewards.append(reward)
            value_functions.append(value)
            log_actions_list.append(log_actions)
            total_reward+=reward.mean()
            # # actor_loss=-((value.detach()-(reward+next_state_value.detach()))*log_actions).mean()
            # # critic_loss=torch.nn.functional.mse_loss(value,reward+next_state_value).mean()
            
            # critic_optim.zero_grad()
            # critic_loss.backward()
            # torch.nn.utils.clip_grad_norm_(critic.parameters(),max_norm=1,norm_type=2)
            # critic_optim.step()

            # actor_optim.zero_grad()
            # actor_loss.backward()
            # torch.nn.utils.clip_grad_norm_(actor.parameters(),max_norm=1,norm_type=2)
            # actor_optim.step()
            # num_iterations+=1
            # print(num_iterations,batch_num)
            # print(reward)
        # total_reward_epoch.append(total_reward)
        # make_spans.append(data.est_end_time.mean())
        print(data.est_end_time.mean(),total_reward,scheduler_critic.get_last_lr()[0],e)
        actor_loss,critic_loss=train_episode(rewards,value_functions,log_actions_list)
        critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(),max_norm=1,norm_type=2)
        critic_optim.step()
        actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(),max_norm=1,norm_type=2)
        actor_optim.step()
    scheduler_actor.step()
    scheduler_critic.step()
torch.save(actor.state_dict(),'scheduler.pth')



    # if scheduler_critic.get_last_lr()[0]>0.00001:
    #     scheduler_critic.step()
    #     scheduler_actor.step()
    # print('total_reward',sum(total_reward_epoch)/len(total_reward_epoch),'make_span',sum(make_spans)/len(make_spans))
        
# print(data.edge_index,data.op_end_time,data.op_machine_map,data.processing_times,op_order)