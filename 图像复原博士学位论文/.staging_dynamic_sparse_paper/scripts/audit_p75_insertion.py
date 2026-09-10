"""Bounded synthetic GPU insertion audit, no optimizer or dataset access."""
import gc, hashlib, json, sys, time
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.reference_nlfe import ReferenceNLFE
from scripts.train_p25_restormer import network


def main():
    out = ROOT / 'results/runs/P75_insertion_v0'
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    try:
        torch.manual_seed(7500)
        path = ROOT / 'third_party/restormer_dpdd_public_v0/single_image_defocus_deblurring.pth'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == '7dce451f33f8f5e0faf7c4e3996e5dcc1bd425ecd1ada99b0f9750e490fd4c9e'
        state = torch.load(path, map_location='cpu', weights_only=True)['params']
        x = torch.rand(4, 3, 128, 128, device='cuda')
        rows = []
        for name, window, pool in [('local', 4, 1), ('nonlocal', 32, 4)]:
            model = network().train()
            model.load_state_dict(state, strict=True)
            torch.manual_seed(7501)
            branch = ReferenceNLFE(192, heads=4, window=window, pool=pool).cuda()
            with torch.no_grad():
                branch.proj.weight.zero_(); branch.proj.bias.zero_()
                branch.ffn[-1].weight.zero_(); branch.ffn[-1].bias.zero_()
            # Module is registered separately on model, hook changes level3 output
            # before both downsampling and the encoder-decoder skip use it.
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                base = model(x).detach().cpu()
            model.add_module('p75_branch', branch)
            handle = model.encoder_level3.register_forward_hook(lambda m, args, y: branch(y))
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize(); tick = time.monotonic()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                pred = model(x)
                loss = (pred.float() - x).abs().mean()
            assert torch.equal(pred.detach().cpu(), base)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            assert branch.proj.weight.grad.abs().sum() > 0
            assert branch.ffn[-1].weight.grad.abs().sum() > 0
            torch.cuda.synchronize()
            rows.append(dict(branch=name, window=window, pool=pool,
                parameters=sum(p.numel() for p in branch.parameters()),
                peak_mib=torch.cuda.max_memory_allocated()/2**20,
                forward_backward_seconds=time.monotonic()-tick,
                exact_initial_output=True, finite_gradients=True))
            assert all(torch.equal(v.cpu(), state[k]) for k, v in model.state_dict().items()
                       if not k.startswith('p75_branch.'))
            handle.remove()
            del model, branch, pred, loss, base
            gc.collect(); torch.cuda.empty_cache()
            assert time.monotonic()-start < 600
        assert rows[0]['parameters'] == rows[1]['parameters']
        result = dict(complete=True, updates=0, synthetic_only=True, rows=rows,
                      seconds=time.monotonic()-start, weights_unchanged=True)
        (out/'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result))
    except Exception as e:
        (out/'failure.json').write_text(json.dumps({'error':repr(e)}), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
