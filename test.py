import torch
from data_generator import data_generator
from state_transition import next_state
from actor import actor
nj=10
nm=10
device=torch.device('mps')
job_dict={'nj':nj,
          'nm':nm
          }
hidden_dim_model=512
test_model=actor(nm+3,512).to(device)
test_model_weights=torch.load(f'scheduler_{nj}_{nm}.pth',weights_only=True)
test_model.load_state_dict(test_model_weights)
num_instances=1000
batches=1
test_model.eval()
test_data=data_generator(nj,nm,1,99,num_instances,batches)
make_span=[]
for data in test_data:
    data.to(device)
    for i in range(nj*nm):
        actions,_=test_model(data)
        data=next_state(data,actions,job_dict)
    make_span.append(data.est_end_time)
    print(data.est_end_time)
print(f'mean reward {sum(make_span)/len(make_span)}')
        
    


