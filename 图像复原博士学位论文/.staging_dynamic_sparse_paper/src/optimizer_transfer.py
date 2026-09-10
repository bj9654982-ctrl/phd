"""Explicit name-based AdamW state transfer; caller must verify source name provenance."""
import copy
import torch

def transfer_adamw(backbone, module, saved, source_names):
    if len(saved['param_groups']) != 1:
        raise ValueError('Only audited single-group source checkpoints are supported')
    group=saved['param_groups'][0]
    ids=group['params'];named=dict(backbone.named_parameters())
    if len(source_names)!=len(ids) or len(set(source_names))!=len(source_names) or set(source_names)!=set(named):
        raise ValueError('Source names must uniquely cover the backbone')
    if set(saved['state'])!=set(ids):raise ValueError('Incomplete source states')
    params=list(named.values());extra=[] if module is None else list(module.parameters())
    if set(map(id,params)) & set(map(id,extra)):raise ValueError('Shared parameters are not supported')
    # Complete validation before building target optimizer. Never mutate saved tensors.
    for name,pid in zip(source_names,ids):
        state=saved['state'][pid];p=named[name]
        for key in ('exp_avg','exp_avg_sq'):
            if state[key].shape!=p.shape or not bool(torch.isfinite(state[key]).all()):
                raise ValueError('Invalid moment for '+name)
        if state['step'].numel()!=1 or not bool(torch.isfinite(state['step']).all()):
            raise ValueError('Invalid step')
    settings={k:copy.deepcopy(v) for k,v in group.items() if k not in ('params','param_names')}
    groups=[{'params':params,**settings}]
    if extra:groups.append({'params':extra,**copy.deepcopy(settings)})
    opt=torch.optim.AdamW(groups)
    remapped={'state':{},'param_groups':opt.state_dict()['param_groups']}
    target_ids=dict(zip(named,remapped['param_groups'][0]['params']))
    for name,pid in zip(source_names,ids):
        remapped['state'][target_ids[name]]=copy.deepcopy(saved['state'][pid])
    opt.load_state_dict(remapped)
    assert all(p not in opt.state for p in extra)
    return opt
