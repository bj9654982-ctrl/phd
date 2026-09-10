"""Read-only checkpoint compatibility audit, no optimizer updates."""
import sys,json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.train_p25_restormer import network
from scripts.analyze_p16_factorial import sha256
from scripts.evaluate_p21_full_image import save_json

def main():
    path=Path('J:/phd/动态稀疏非局部去模糊论文/results/runs/P26_restormer_continue10000_v0/step_10000.pt')
    ckpt=torch.load(path,map_location='cpu',weights_only=False)
    model=network().cpu();model.load_state_dict(ckpt['model'],strict=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    optimizer.load_state_dict(ckpt['optimizer'])
    steps=[];mapping=[]
    for name,p in model.named_parameters():
        state=optimizer.state[p]
        assert set(('step','exp_avg','exp_avg_sq'))<=set(state)
        for key in ('exp_avg','exp_avg_sq'):
            assert state[key].shape==p.shape and bool(torch.isfinite(state[key]).all())
        steps.append(int(state['step'].item()));mapping.append({'name':name,'shape':list(p.shape)})
    assert set(steps)=={10000} and ckpt['step']==10000
    result={'complete':True,'checkpoint_hash':sha256(path),'checkpoint_step':ckpt['step'],
        'parameter_count':len(mapping),'optimizer_steps':sorted(set(steps)),
        'groups':[{k:v for k,v in g.items() if k!='params'} for g in optimizer.param_groups],
        'mapping':mapping,'updates':0,'official_test_accessed':False,
        'mapping_basis':'train_p25_restormer.py uses model.parameters() for construction and resume; shapes checked'}
    out=ROOT/'results/runs/P42_optimizer_audit_v0';out.mkdir(parents=True,exist_ok=False)
    save_json(out/'summary.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='mapping'},indent=2))
if __name__=='__main__':main()
