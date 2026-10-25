"""Function-preserving residual-depth expansion using only our competition weights."""
import re
import torch
from model import ImageModel

STUDENT='resnet34_graft'
def initialize(source_state):
    student=ImageModel(STUDENT)
    state=student.state_dict();mapping={}
    for key,target in state.items():
        source_key=key
        if key not in source_state:
            match=re.match(r'network\.layer([1-4])\.(\d+)\.(.+)',key)
            if not match or int(match.group(2))<2:raise KeyError(key)
            source_key=f'network.layer{match.group(1)}.1.{match.group(3)}'
        value=source_state[source_key]
        assert value.shape==target.shape,(key,source_key)
        state[key]=value.detach().clone();mapping[key]=source_key
    student.load_state_dict(state)
    prefixes=[]
    with torch.no_grad():
        for layer,count in [(1,3),(2,4),(3,6),(4,3)]:
            blocks=getattr(student.network,f'layer{layer}')
            for index in range(2,count):
                prefixes.append(f'network.layer{layer}.{index}.')
                blocks[index].bn2.weight.zero_();blocks[index].bn2.bias.zero_()
    assert len(prefixes)==8
    for p in student.parameters():p.requires_grad_(False)
    for name,p in student.named_parameters():
        if name.startswith(tuple(prefixes)) or name.startswith('network.fc.'):
            p.requires_grad_(True)
    return student,prefixes,mapping

def training_mode(student,prefixes):
    student.eval()
    for prefix in prefixes:
        _,layer,index,_=prefix.split('.')
        getattr(student.network,layer)[int(index)].train()
    student.network.fc.train()

def frozen_state(student,prefixes):
    return {k:v.detach().cpu().clone() for k,v in student.state_dict().items()
        if not k.startswith(tuple(prefixes)) and not k.startswith('network.fc.')}

def assert_frozen(student,reference):
    assert all(torch.equal(v,student.state_dict()[k].detach().cpu()) for k,v in reference.items()),'An inherited frozen block changed'
