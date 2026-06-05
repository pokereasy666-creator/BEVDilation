"""Offline CPU validation of InstanceGuidedFusion.

Runs without mmcv/CUDA by using the pure-PyTorch deformable-attention path.
Checks the things that are easy to get wrong in this integration:
  1. shape contract: (B, Cin, H, W) -> (B, Cin, H, W) and (B, ncls, H, W)
  2. BEV orientation / selection: the top-K picks the cell at the heatmap peak,
     and the (x, y) deformable reference for that instance maps to the SAME
     (row, col), i.e. no transpose between the heatmap and the sampling grid
  3. identity at init: zero-init exit => out == x exactly before any training
  4. gradient flow: loss on the output backprops into IGF params AND into x
  5. heatmap-vs-feature decoupling: the aux heatmap head input is detached, so
     the heatmap loss does NOT produce gradients on the main feature path
"""
import os
import sys
import torch
import torch.nn.functional as F

# Direct file import (no package import): keeps this a torch-only correctness
# gate that runs before the full mmdet3d/mmcv environment is set up. Importing
# `mmdet3d.models.fusion_layers` would pull in the whole mmcv/mmdet chain.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'mmdet3d', 'models', 'fusion_layers'))
from instance_guided_fusion import InstanceGuidedFusion

torch.manual_seed(0)

B, Cin, S, NCLS, K = 2, 256, 16, 10, 12  # small bev_size for speed
igf = InstanceGuidedFusion(
    in_channels=Cin, inner_channels=64, bev_size=S, num_classes=NCLS,
    instance_num=K, n_points=8, num_context_layers=2,
    use_pytorch_deform=True).train()

# ---- 3. identity at init -------------------------------------------------- #
x = torch.randn(B, Cin, S, S)
out, ins_hm = igf(x)
assert out.shape == (B, Cin, S, S), out.shape
assert ins_hm.shape == (B, NCLS, S, S), ins_hm.shape
id_err = (out - x).abs().max().item()
print(f'[1/5] shapes ok: out {tuple(out.shape)}, hm {tuple(ins_hm.shape)}')
print(f'[2/5] identity-at-init max|out-x| = {id_err:.3e}',
      '-> PASS' if id_err < 1e-5 else '-> FAIL')
assert id_err < 1e-5

# ---- 2. orientation / selection ------------------------------------------ #
# Drive the selection by overriding the heatmap with a known single peak at
# (row=r0, col=c0) on class 0; verify the recovered reference (x,y) == (c0,r0).
r0, c0 = 3, 11
captured = {}
real_inst_att = igf.instance_att.forward

def spy(query_feats, reference_points, key_coords, scene_feats):
    captured['ref'] = reference_points.detach().clone()
    return real_inst_att(query_feats, reference_points, key_coords, scene_feats)

igf.instance_att.forward = spy

fake_logits = torch.full((B, NCLS, S, S), -10.0)
fake_logits[:, 0, r0, c0] = 10.0
orig_predict = igf._predict_heatmap
igf._predict_heatmap = lambda feat: fake_logits  # logits ignore feat here
_ = igf(x)
igf._predict_heatmap = orig_predict
igf.instance_att.forward = real_inst_att

ref = captured['ref']                       # (B, K, 2) normalized (x, y)
top_x = ref[0, 0, 0].item() * S             # de-normalize
top_y = ref[0, 0, 1].item() * S
# cell centre is +0.5; peak cell (r0,c0) -> x=c0+0.5, y=r0+0.5
ok_x = abs(top_x - (c0 + 0.5)) < 1e-4
ok_y = abs(top_y - (r0 + 0.5)) < 1e-4
print(f'[3/5] peak at (row={r0}, col={c0}) -> ref (x={top_x:.2f}, y={top_y:.2f}); '
      f'expect (x={c0 + 0.5}, y={r0 + 0.5})',
      '-> PASS' if (ok_x and ok_y) else '-> FAIL')
assert ok_x and ok_y, 'BEV orientation/selection mismatch (transpose bug)'

# ---- 4. gradient flow (output -> IGF params and -> input) ----------------- #
igf.zero_grad()
x2 = torch.randn(B, Cin, S, S, requires_grad=True)
out2, hm2 = igf(x2)
# perturb exit last layer so the residual is non-trivial (init is zero)
with torch.no_grad():
    igf.exit[-1].weight.add_(torch.randn_like(igf.exit[-1].weight) * 0.01)
out2, hm2 = igf(x2)
loss_main = out2.pow(2).mean()
loss_main.backward()
g_input = x2.grad.abs().sum().item()
g_entry = igf.entry[0].weight.grad.abs().sum().item()
g_i2s = igf.instance_to_scene.attn.in_proj_weight.grad.abs().sum().item()
g_deform = (igf.instance_att.layers[0].cross_attn.value_proj.weight.grad
            .abs().sum().item())
print(f'[4/5] grad to input={g_input:.3e}, entry={g_entry:.3e}, '
      f'I2S_attn={g_i2s:.3e}, deform_value={g_deform:.3e}',
      '-> PASS' if min(g_input, g_entry, g_i2s, g_deform) > 0 else '-> FAIL')
assert min(g_input, g_entry, g_i2s, g_deform) > 0

# ---- 5. heatmap head input is detached from the main feature -------------- #
igf.zero_grad()
x3 = torch.randn(B, Cin, S, S, requires_grad=True)
out3, hm3 = igf(x3)
heat_target = torch.zeros_like(hm3)
heat_target[:, 0, r0, c0] = 1.0
loss_hm = F.binary_cross_entropy_with_logits(hm3, heat_target)
loss_hm.backward()
# entry feeds BOTH the (detached) heatmap branch and the (live) scene branch.
# With ONLY the heatmap loss, grad reaches entry only if the detach leaked.
g_entry_from_hm = (0.0 if igf.entry[0].weight.grad is None
                   else igf.entry[0].weight.grad.abs().sum().item())
g_hmhead = igf.heatmap_head_3.weight.grad.abs().sum().item()
print(f'[5/5] heatmap-only loss: grad to heatmap_head={g_hmhead:.3e} (>0), '
      f'grad to entry={g_entry_from_hm:.3e} (==0 expected)',
      '-> PASS' if (g_hmhead > 0 and g_entry_from_hm == 0) else '-> FAIL')
assert g_hmhead > 0 and g_entry_from_hm == 0

print('\nALL CHECKS PASSED')
