# ---------------------------------------------------------------------------
# Instance-Guided Fusion (IGF), adapted from IS-Fusion (Yin et al., CVPR 2024)
# for use as a drop-in BEV-feature refinement block on top of BEVDilation's
# dense LiDAR BEV feature (the output of the 2D dense backbone, i.e. B'_F).
#
# Design notes (read before editing):
#   * This is the "variant A" / LiDAR-centric integration agreed in the design
#     discussion: IGF's deformable context-aggregation samples the SAME LiDAR
#     BEV feature that is fed in. No image-derived feature is injected here, so
#     BEVDilation's LiDAR-centric regression regime is preserved.
#   * Convention: the input BEV tensor x is (B, C, H, W) with dim -2 = y (row)
#     and dim -1 = x (col), matching mmdet3d's heatmap target
#     (num_classes, feature_map_size[1]=H=y, feature_map_size[0]=W=x). All
#     internal flattening is row-major (n = y*W + x). Deformable-attention
#     reference points are (x_norm=col/W, y_norm=row/H) with spatial_shapes
#     (H, W); this matches mmcv's offset_normalizer = (W, H) and grid_sample's
#     x=width convention. There are NO axis permutes (unlike the original
#     IS-Fusion code, whose permutes are an artifact of its HSF feed and are
#     not semantically required here).
#   * IGF runs at `inner_channels` (default 128, the width IS-Fusion actually
#     used) to keep the dense instance-to-scene attention memory bounded; the
#     output is projected back to `in_channels` and added as a residual whose
#     last conv is zero-initialised, so at init IGF == identity. This is
#     deliberate: BEVDilation's BEV feature is already strong, so IGF must not
#     perturb it before it has learned anything.
#   * The auxiliary instance heatmap (logits, (B, num_classes, H, W),
#     x-oriented) is RETURNED for supervision by the detection head against the
#     same Gaussian target as the head's own dense heatmap. The head loss is
#     unchanged except for an additive, optional `ins_heatmap` term.
# ---------------------------------------------------------------------------

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Pure-PyTorch multi-scale deformable attention (used for CPU testing and as a
# fallback). Numerically identical to mmcv.ops.multi_scale_deformable_attn_pytorch.
# --------------------------------------------------------------------------- #
def ms_deform_attn_pytorch(value, value_spatial_shapes, sampling_locations,
                           attention_weights):
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, _, num_levels, num_points, _ = sampling_locations.shape
    split_sizes = [int(H) * int(W) for H, W in value_spatial_shapes]
    value_list = value.split(split_sizes, dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H, W) in enumerate(value_spatial_shapes):
        H, W = int(H), int(W)
        # (bs, H*W, nh, c) -> (bs*nh, c, H, W)
        value_l = (value_list[level].flatten(2).transpose(1, 2)
                   .reshape(bs * num_heads, embed_dims, H, W))
        # (bs, nq, nh, np, 2) -> (bs*nh, nq, np, 2)
        sampling_grid_l = (sampling_grids[:, :, :, level]
                           .transpose(1, 2).flatten(0, 1))
        sampling_value_l = F.grid_sample(
            value_l, sampling_grid_l, mode='bilinear',
            padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l)
    # (bs, nq, nh, nl, np) -> (bs*nh, 1, nq, nl*np)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points)
    output = ((torch.stack(sampling_value_list, dim=-2).flatten(-2)
               * attention_weights).sum(-1)
              .view(bs, num_heads * embed_dims, num_queries))
    return output.transpose(1, 2).contiguous()


def _load_cuda_deform_fn():
    """Lazily import the mmcv-_ext-backed CUDA deform function (their env)."""
    from .ms_deform_attn_function import MultiScaleDeformableAttnFunction_fp32
    return MultiScaleDeformableAttnFunction_fp32


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def conv_bn_relu(in_c, out_c, k=3, p=1):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=k, padding=p, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


def _get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class PositionEmbeddingLearned(nn.Module):
    """Learned absolute position embedding for (B, N, in_dim) coordinates."""

    def __init__(self, input_channel, num_pos_feats):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, 1),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, 1),
        )

    def forward(self, coords):  # coords: (B, N, in_dim)
        coords = coords.transpose(1, 2).contiguous()  # (B, in_dim, N)
        return self.head(coords)                       # (B, num_pos_feats, N)


class MSDeformAttn(nn.Module):
    """Single-level deformable attention (copied math from IS-Fusion / DETR)."""

    def __init__(self, d_model=256, n_levels=1, n_heads=8, n_points=16,
                 use_pytorch=False):
        super().__init__()
        assert d_model % n_heads == 0, (d_model, n_heads)
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.im2col_step = 64
        self.use_pytorch = use_pytorch
        self._cuda_fn = None

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]) \
            .view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        nn.init.constant_(self.attention_weights.weight.data, 0.)
        nn.init.constant_(self.attention_weights.bias.data, 0.)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten,
                input_spatial_shapes, input_level_start_index):
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        value = self.value_proj(input_flatten)
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(
            N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points)
        # reference_points: (N, Len_q, n_levels, 2) in (x, y), range [0, 1]
        offset_normalizer = torch.stack(
            [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
        sampling_locations = (
            reference_points[:, :, None, :, None, :]
            + sampling_offsets / offset_normalizer[None, None, None, :, None, :])

        if self.use_pytorch:
            output = ms_deform_attn_pytorch(
                value, input_spatial_shapes, sampling_locations, attention_weights)
        else:
            if self._cuda_fn is None:
                self._cuda_fn = _load_cuda_deform_fn()
            output = self._cuda_fn.apply(
                value, input_spatial_shapes, input_level_start_index,
                sampling_locations, attention_weights, self.im2col_step)
        return self.output_proj(output)


class DeformableDecoderLayer(nn.Module):
    """Self-attn over instances + deformable cross-attn into the scene + FFN."""

    def __init__(self, d_model=128, d_ffn=128, dropout=0.1, n_levels=1,
                 n_heads=8, n_points=16, use_pytorch=False):
        super().__init__()
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points,
                                       use_pytorch=use_pytorch)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def _with_pos(t, pos):
        return t if pos is None else t + pos

    def _ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        return self.norm3(tgt)

    def forward(self, tgt, query_pos, reference_points, src,
                src_spatial_shapes, level_start_index):
        # tgt: (B, K, C); query_pos: (B, K, C); reference_points: (B, K, 1, 2)
        q = k = self._with_pos(tgt, query_pos)
        tgt2 = self.self_attn(
            q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1),
            need_weights=False)[0].transpose(0, 1)
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        tgt2 = self.cross_attn(
            self._with_pos(tgt, query_pos), reference_points, src,
            src_spatial_shapes, level_start_index)
        tgt = self.norm1(tgt + self.dropout1(tgt2))
        return self._ffn(tgt)


class InstanceContextAggregation(nn.Module):
    """f_agg: inter-instance self-attention + deformable context aggregation.

    Samples context from `scene_feats` (the LiDAR BEV feature) — variant A.
    """

    def __init__(self, num_layers=2, embed_dims=128, bev_h=180, bev_w=180,
                 n_points=16, n_heads=8, dropout=0.1, use_pytorch=False):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        layer = DeformableDecoderLayer(
            d_model=embed_dims, d_ffn=embed_dims, dropout=dropout,
            n_levels=1, n_heads=n_heads, n_points=n_points, use_pytorch=use_pytorch)
        self.layers = _get_clones(layer, num_layers)
        self.query_pos_embed = PositionEmbeddingLearned(2, embed_dims)
        self.key_pos_embed = PositionEmbeddingLearned(2, embed_dims)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

    def forward(self, query_feats, reference_points, key_coords, scene_feats):
        """
        query_feats:      (B, C, K)        instance features
        reference_points: (B, K, 2)        (x_norm, y_norm) in [0, 1]
        key_coords:       (B, H*W, 2)       (x_norm, y_norm) for every BEV cell
        scene_feats:      (B, C, H, W)      context source (LiDAR BEV)
        returns:          (B, C, K)
        """
        B, C, H, W = scene_feats.shape
        key_pos = self.key_pos_embed(key_coords).permute(0, 2, 1)  # (B, H*W, C)
        # row-major flatten (n = y*W + x), matching key_coords ordering
        src = scene_feats.flatten(2).transpose(1, 2) + key_pos      # (B, H*W, C)

        output = query_feats.transpose(1, 2)                        # (B, K, C)
        query_pos = self.query_pos_embed(reference_points).permute(0, 2, 1)  # (B, K, C)

        spatial_shapes = torch.as_tensor(
            [(H, W)], dtype=torch.long, device=src.device)
        level_start_index = torch.zeros(
            (1,), dtype=torch.long, device=src.device)

        ref = reference_points[:, :, None, :]                       # (B, K, 1, 2)
        for layer in self.layers:
            output = layer(output, query_pos, ref, src,
                           spatial_shapes, level_start_index)
        return output.transpose(1, 2)                               # (B, C, K)


class InstanceToScene(nn.Module):
    """f_I2S: every BEV cell attends to the K instances, then a per-channel
    spatial mixing writes the instance information back. Returns the UPDATE
    only (the residual is added at full width by the parent module)."""

    def __init__(self, d_model=128, n_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query_value, instance_feats, scene_feats, B, H, W):
        """
        query_value:    (B, C, H*W)   value carried by each BEV cell (conv_ins)
        instance_feats: (B, C, K)      enriched instance features
        scene_feats:    (B, C, H, W)   scene used as the spatial-mix query
        returns:        (B, C, H, W)   update (no residual)
        """
        q = query_value.permute(2, 0, 1)        # (H*W, B, C)
        k = instance_feats.permute(2, 0, 1)      # (K, B, C)
        q2 = self.attn(q, k, k, need_weights=False)[0]               # (H*W, B, C)
        q = self.norm(q + self.dropout(q2)).permute(1, 2, 0)        # (B, C, H*W)
        query_ins = q.reshape(B, q.shape[1], H, W)                  # row-major

        attn_w = torch.matmul(scene_feats, query_ins.transpose(2, 3))  # (B,C,H,W)
        attn_w = F.softmax(attn_w, dim=-1)
        attended = torch.matmul(attn_w, query_ins)                  # (B, C, H, W)
        return attended


# --------------------------------------------------------------------------- #
# Top-level module
# --------------------------------------------------------------------------- #
class InstanceGuidedFusion(nn.Module):
    """Instance-Guided Fusion as a BEV-feature refinement block.

    forward(x: (B, in_channels, H, W)) ->
        out:   (B, in_channels, H, W)   instance-aware BEV (identity at init)
        ins_hm:(B, num_classes, H, W)   instance heatmap LOGITS (x-oriented)
    """

    def __init__(self,
                 in_channels=256,
                 inner_channels=128,
                 bev_size=180,
                 num_classes=10,
                 instance_num=200,
                 n_points=16,
                 num_context_layers=2,
                 n_heads=8,
                 nms_kernel_size=3,
                 dropout=0.1,
                 use_pytorch_deform=False):
        super().__init__()
        if isinstance(bev_size, (tuple, list)):
            self.bev_h, self.bev_w = int(bev_size[0]), int(bev_size[1])
        else:
            self.bev_h = self.bev_w = int(bev_size)
        self.num_classes = num_classes
        self.instance_num = instance_num
        self.nms_kernel_size = nms_kernel_size
        # Step-9 redundancy diagnostic: relative residual energy of the last
        # forward (||exit(update)|| / (||x|| + eps)), updated under self.training
        # on the first sample only. Surfaced by the detector under a non-loss key.
        self._last_residual_ratio = 0.0

        c = inner_channels
        self.entry = conv_bn_relu(in_channels, c)

        self.conv_heatmap = conv_bn_relu(c, c)
        self.heatmap_head_1 = conv_bn_relu(c, c // 2)
        self.heatmap_head_2 = conv_bn_relu(c // 2, c // 2)
        self.heatmap_head_3 = nn.Conv2d(c // 2, num_classes, 3, 1, 1)

        self.conv_scene = conv_bn_relu(c, c)
        self.conv_ins = conv_bn_relu(c, c)

        self.instance_att = InstanceContextAggregation(
            num_layers=num_context_layers, embed_dims=c,
            bev_h=self.bev_h, bev_w=self.bev_w, n_points=n_points,
            n_heads=n_heads, dropout=dropout, use_pytorch=use_pytorch_deform)
        self.instance_to_scene = InstanceToScene(
            d_model=c, n_heads=n_heads, dropout=dropout)

        # exit projection back to in_channels; last conv zero-init -> identity
        self.exit = nn.Sequential(
            nn.Conv2d(c, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1, bias=True),
        )
        nn.init.constant_(self.exit[-1].weight.data, 0.)
        nn.init.constant_(self.exit[-1].bias.data, 0.)

        # cell-centre (x=col, y=row) grid, row-major (n = y*W + x)
        ys, xs = torch.meshgrid(
            torch.arange(self.bev_h, dtype=torch.float32),
            torch.arange(self.bev_w, dtype=torch.float32),
            indexing='ij')
        coords = torch.stack([xs + 0.5, ys + 0.5], dim=-1).reshape(1, -1, 2)
        self.register_buffer('bev_xy', coords, persistent=False)

    def _predict_heatmap(self, feat):
        h = self.conv_heatmap(feat)
        h = self.heatmap_head_1(h)
        h = self.heatmap_head_2(h)
        return self.heatmap_head_3(h)  # logits (B, num_classes, H, W)

    def _select_topk(self, heatmap_logits):
        """Local-max NMS + top-K over (class, position). Returns flat indices
        into H*W (row-major) of shape (B, K)."""
        B = heatmap_logits.shape[0]
        H, W = self.bev_h, self.bev_w
        heat = heatmap_logits.detach().sigmoid()
        pad = self.nms_kernel_size // 2
        local_max = torch.zeros_like(heat)
        inner = F.max_pool2d(heat, self.nms_kernel_size, stride=1, padding=0)
        local_max[:, :, pad:(-pad), pad:(-pad)] = inner
        if self.num_classes == 10:  # nuScenes: keep peaky pedestrian/cone
            local_max[:, 8] = F.max_pool2d(heat[:, 8], 1, stride=1, padding=0)
            local_max[:, 9] = F.max_pool2d(heat[:, 9], 1, stride=1, padding=0)
        heat = heat * (heat == local_max)
        heat = heat.view(B, self.num_classes, -1)               # (B, nc, H*W)
        top = heat.view(B, -1).argsort(dim=-1, descending=True)[..., :self.instance_num]
        top_index = top % heat.shape[-1]                        # (B, K) into H*W
        return top_index

    def forward(self, x):
        B, _, H, W = x.shape
        assert (H, W) == (self.bev_h, self.bev_w), \
            f'IGF expected BEV {(self.bev_h, self.bev_w)}, got {(H, W)}'

        xr = self.entry(x)                                       # (B, c, H, W)

        # auxiliary instance heatmap (detached input: aux loss must not perturb
        # the main feature, only train the heatmap head) -- as in IS-Fusion
        ins_hm = self._predict_heatmap(xr.detach())             # (B, nc, H, W)
        top_index = self._select_topk(ins_hm)                   # (B, K)

        bev_xy = self.bev_xy.to(x.device).repeat(B, 1, 1)       # (B, H*W, 2)
        ref_xy = bev_xy.gather(
            1, top_index[:, :, None].expand(-1, -1, 2))         # (B, K, 2)
        # normalise to [0, 1]
        norm = torch.tensor([W, H], dtype=x.dtype, device=x.device)
        ref_xy_n = ref_xy / norm
        key_xy_n = bev_xy / norm

        x_scene = self.conv_scene(xr)                           # (B, c, H, W)
        scene_flat = x_scene.flatten(2)                         # (B, c, H*W)
        x_ins = scene_flat.gather(
            2, top_index[:, None, :].expand(-1, x_scene.shape[1], -1))  # (B, c, K)

        x_ins = self.instance_att(x_ins, ref_xy_n, key_xy_n, x_scene)   # (B, c, K)

        x_val = self.conv_ins(xr).flatten(2)                    # (B, c, H*W)
        update = self.instance_to_scene(x_val, x_ins, x_scene, B, H, W)  # (B, c, H, W)

        delta = self.exit(update)                               # zero at init
        if self.training:
            # cheap redundancy diagnostic on the first sample only (detached)
            with torch.no_grad():
                ratio = delta[:1].norm() / (x[:1].norm() + 1e-6)
                self._last_residual_ratio = float(ratio.detach())
        out = x + delta                                         # identity at init
        return out, ins_hm
