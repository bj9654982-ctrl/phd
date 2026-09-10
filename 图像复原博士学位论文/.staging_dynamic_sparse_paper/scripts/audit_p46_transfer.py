"""Real checkpoint transfer audit; no data and no optimizer updates."""
import sys,json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.train_p25_restormer import network
from scripts.analyze_p16_factorial import sha256
from scripts.evaluate_p21_full_image import save_json
from src.models.pooled_context import PooledContext
from src.optimizer_transfer import transfer_adamw

def main():
    path=Path('J:/phd/动态稀疏非局部去模糊论文/results/runs/P26_restormer_continue10000_v0/step_10000.pt')
    digest=sha256(path)
    assert digest=='47052b4b47d437d736ee1cb637ba744a22d86744d93abfd226382e7e8ef2d3c2'
    ckpt=torch.load(path,map_location='cpu',weights_only=False)
    model=network().cpu();model.load_state_dict(ckpt['model'],strict=True)
    # network() verifies pinned architecture/config hashes; original trainer uses model.parameters().
    names=[n for n,p in model.named_parameters()];module=PooledContext()
    opt=transfer_adamw(model,module,ckpt['optimizer'],names)
    ids=ckpt['optimizer']['param_groups'][0]['params']
    for (name,p),pid in zip(model.named_parameters(),ids):
        source=ckpt['optimizer']['state'][pid]
        for key in ('step','exp_avg','exp_avg_sq'):
            assert torch.equal(opt.state[p][key].cpu(),source[key])
            assert opt.state[p][key].data_ptr()!=source[key].data_ptr()
        assert int(opt.state[p]['step'].item())==10000
    assert all(p not in opt.state for p in module.parameters())
    settings={k:v for k,v in ckpt['optimizer']['param_groups'][0].items() if k not in ('params','param_names')}
    assert all(all(g[k]==v for k,v in settings.items()) for g in opt.param_groups)
    assert all(torch.equal(v,ckpt['model'][k]) for k,v in model.state_dict().items())
    assert sha256(path)==digest
    out=ROOT/'results/runs/P46_transfer_audit_v0';out.mkdir(parents=True,exist_ok=False)
    result={'complete':True,'backbone_parameter_tensors':len(names),'module_parameter_tensors':len(list(module.parameters())),
            'exact_moments_and_steps':True,'new_module_state_empty':True,'source_file_unchanged':True,
            'independent_state_storage':True,'hyperparameters_equal':True,'updates':0,'data_accessed':False,
            'checkpoint_hash':digest,'transfer_hash':sha256(ROOT/'src/optimizer_transfer.py')}
    save_json(out/'summary.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
