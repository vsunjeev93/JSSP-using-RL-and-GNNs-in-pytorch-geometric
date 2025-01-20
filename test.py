import torch
from data_generator import data_generator
from state_transition import next_state
from actor import actor
import argparse
parser=argparse.ArgumentParser('specify the number of jobs and machines')
parser.add_argument('--nj',type=int,default=10)
parser.add_argument('--nm',type=int,default=10)
parser.add_argument('--seed',type=int,default=42)
args=parser.parse_args()
device = torch.device("mps")
job_dict = {"nj": args.nj, "nm": args.nm}
hidden_dim_model = 512
test_model = actor(args.nm + 3, 512).to(device)
test_model_weights = torch.load(f"scheduler_{args.nj}_{args.nm}.pth", weights_only=True)
test_model.load_state_dict(test_model_weights)
num_instances = 100
batches = 1
test_model.eval()
test_data = data_generator(args.nj, args.nm, 1, 99, num_instances, batches,seed=args.seed)
make_span = []
for data in test_data:
    data.to(device)
    for i in range(args.nj * args.nm):
        actions, _ = test_model(data)
        data = next_state(data, actions, job_dict)
    make_span.append(data.est_end_time)
    print(data.est_end_time)
print(f"mean reward {sum(make_span)/len(make_span)}")
