from data_generator import data_generator
from actor import actor
from critic import critic
import torch
import torch_geometric.transforms as transform
from state_transition import next_state
import argparse

# initializing the job parameters
parser=argparse.ArgumentParser('specify the number of jobs and machines')
parser.add_argument('--nj',type=int,default=10)
parser.add_argument('--nm',type=int,default=10)
parser.add_argument('--seed',type=int,default=42)
args=parser.parse_args()
param_dict = {
    "nj": args.nj,
    "nm": args.nm,
    "low": 1,
    "high": 99,
    "instances": 300,
    "batch_size": 10,
}
# setting device and creating instances of actor and critic function approximators
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
input_features = param_dict["nm"] + 3
actor = actor(input_features, 512)
actor.to(device)
critic = critic(input_features, 512)
critic.to(device)
torch.manual_seed(args.seed)
# setting optimizer, learning rate and other training parameters
LR = 0.0001
actor_optim = torch.optim.Adam(actor.parameters(), lr=LR)
critic_optim = torch.optim.Adam(critic.parameters(), lr=LR)
scheduler_actor = torch.optim.lr_scheduler.StepLR(actor_optim, step_size=20, gamma=0.9)
scheduler_critic = torch.optim.lr_scheduler.StepLR(
    critic_optim, step_size=20, gamma=0.9
)
epoch = 200


def train_episode(rewards, value_functions, log_actions):
    """
    this function takes in a list of rewards, values and log actions
    in an episode and calculates the actor and critic losses
    """
    R = 0
    gamma = 1
    returns = []
    actor_loss = []
    critic_loss = []
    for r in rewards[::-1]:
        R = r + R * gamma
        returns.insert(0, R)
    for R, value, log_prob in zip(returns, value_functions, log_actions):
        advantage = R - value
        actor_loss.append(advantage.detach() * log_prob * -1)
        critic_loss.append(torch.nn.functional.mse_loss(R, value))
    actor_loss = torch.stack(actor_loss)
    critic_loss = torch.stack(critic_loss)
    return actor_loss.sum().mean(), critic_loss.sum().mean()


# training starts here
for e in range(epoch):
    loader = data_generator(*param_dict.values(),seed=args.seed)
    for data in loader:
        data = data.to(device)
        total_reward = 0
        rewards = []
        value_functions = []
        log_actions_list = []
        for i in range(param_dict["nj"] * param_dict["nm"]):
            actions, log_actions = actor(data)
            value = critic(data)
            est_makespan = data.est_end_time
            data = next_state(data, actions, param_dict)
            reward = est_makespan - data.est_end_time
            rewards.append(reward)
            value_functions.append(value)
            log_actions_list.append(log_actions)
            total_reward += reward.mean()
        print(
            data.est_end_time.mean(), total_reward, scheduler_critic.get_last_lr()[0], e
        )
        actor_loss, critic_loss = train_episode(
            rewards, value_functions, log_actions_list
        )
        critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1, norm_type=2)
        critic_optim.step()
        actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1, norm_type=2)
        actor_optim.step()
    scheduler_actor.step()
    scheduler_critic.step()
torch.save(actor.state_dict(), f"scheduler_{param_dict['nj']}_{param_dict['nm']}.pth")
