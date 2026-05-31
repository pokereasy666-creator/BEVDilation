# Copyright (c) OpenMMLab. All rights reserved.
"""Diagnostic 1 (P_fg stability) -- analyze stage.

Reads the per-frame ``.npz`` dumps produced by ``diag1_capture.py`` and measures
the temporal stability of the foreground logit field ``z_fg``. This decides
whether Temporal Foreground Propagation (TFP) targets a real problem: if the
foreground logits are already stable on the hard stratum (sparse / static
objects), the TFP stability case is weak.

This script is numpy-only and runs anywhere (no torch / mmcv / GPU).

Two metrics
-----------
PRIMARY (decision-relevant) -- center-based, object-tracked. Tracks GT boxes
across consecutive frames with conservative velocity-compensated matching, then
samples ``z_fg`` at each tracked object's center via bilinear interpolation in
that frame's own grid. This avoids the spatial-quantization artifact of warping
the whole grid (rounding warped positions to 0.6 m cells injects boundary noise
that dwarfs real prediction instability).

SECONDARY (diagnostic, artifact-contaminated, baseline-subtracted) -- warps each
frame's ``z_fg`` into the scene's first-frame grid and measures per-cell
temporal std at foreground vs background cells; the background std estimates the
warp-quantization floor that is subtracted off.

FRAME-VERIFICATION GATE (``--verify-frame``) -- the primary metric assumes the
stored ``gt_boxes`` xy live in the same frame as the ``z_fg`` grid. This gate
checks that against the model's learned GT-box / foreground alignment and
reports any systematic offset; run it FIRST on real dumps.

Geometry matches voxel_generation.py's BEV mapping: grid 180x180, downstride 8,
voxel_size 0.075 -> 0.6 m cells, point_cloud_range min -54 for x and y. Cell
[row, col] -> physical x = (col+0.5)*0.6 - 54, y = (row+0.5)*0.6 - 54.
"""
import argparse
import glob
import json
import math
import os.path as osp
import warnings
from collections import defaultdict

import numpy as np

# ---- config constants ----
GRID = 180
CELL = 0.6                  # downstride(8) * voxel_size(0.075)
PC_MIN = -54.0              # point_cloud_range min for x and y
# Empirically-verified row correction (paired with the rot270 fix in load_z_fg):
# after rot270, GT box centers still land 2 cells low in the row direction. A
# row-offset sweep over ~27,500 GT centers across ~1000 frames (all scenes)
# found mean z_fg peaks at +2 rows (+0.027) vs the unshifted -0.792 at +0
# (+1 -> -0.258, +3 -> -0.423); the column profile is symmetric (no col offset).
ROW_OFFSET = 2
SPARSE_MAX_PTS = 10
STATIC_MAX_SPEED = 0.5      # m/s
MIN_TRACK_FRAMES = 3
MATCH_GATE_M = 2.0          # m, residual after motion compensation
STD_THRESH = 0.30           # decision threshold on sparse_static mean std
FG_LOGIT_THR = math.log(0.4 / 0.6)   # logit for sigmoid == fg_thr(0.4) ~ -0.4055
VERIFY_SHIFT = 4            # +/- cell range searched in the frame-check
VERIFY_IOU_OK = 0.20
VERIFY_IOU_BAD = 0.10

# stratification buckets, fixed order for reporting
BUCKETS = ('sparse_static', 'sparse_dynamic', 'dense_static', 'dense_dynamic')
# fg_fraction histogram edges: [0-.1)(.1-.3)(.3-.5)(.5-.7)(.7-.9)(.9-1]
FG_FRACTION_BINS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]


# --------------------------------------------------------------------------- #
# Geometry helpers (verified against voxel_generation.py's BEV mapping)
# --------------------------------------------------------------------------- #
def quat_to_rotmat(q):
    """Quaternion q = [w, x, y, z] -> 3x3 rotation matrix."""
    w, x, y, z = [float(v) for v in q]
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ])


def ego_xy_to_global(xy, l2e_t, l2e_R, e2g_t, e2g_R):
    """Box-frame xy (...,2) -> global xy. Append z=0, lidar->ego then ego->global."""
    xy = np.atleast_2d(np.asarray(xy, dtype=np.float64))
    p = np.concatenate([xy, np.zeros((xy.shape[0], 1))], axis=1)
    p = p @ l2e_R.T + l2e_t
    p = p @ e2g_R.T + e2g_t
    return p[:, :2]


def ego_vec_to_global(v_xy, l2e_R, e2g_R):
    """Velocity vector (...,2) -> global, rotation only (no translation)."""
    v = np.atleast_2d(np.asarray(v_xy, dtype=np.float64))
    v = np.concatenate([v, np.zeros((v.shape[0], 1))], axis=1)
    v = v @ l2e_R.T
    v = v @ e2g_R.T
    return v[:, :2]


def global_xy_to_ego_cell(xy, l2e_t, l2e_R, e2g_t, e2g_R):
    """Inverse of ego_xy_to_global, then to (row, col) cell coords (...,2)."""
    xy = np.atleast_2d(np.asarray(xy, dtype=np.float64))
    p = np.concatenate([xy, np.zeros((xy.shape[0], 1))], axis=1)
    p = (p - e2g_t) @ e2g_R
    p = (p - l2e_t) @ l2e_R
    col = (p[:, 0] - PC_MIN) / CELL - 0.5
    row = (p[:, 1] - PC_MIN) / CELL - 0.5 + ROW_OFFSET
    return np.stack([row, col], axis=1)


def phys_to_cell(x, y):
    """In-frame physical (x, y) -> (row, col) float (primary in-frame sampling).

    Adds the empirical +ROW_OFFSET so GT centers land on the foreground peak in
    the rot270-corrected z_fg frame.
    """
    col = (np.asarray(x, dtype=np.float64) - PC_MIN) / CELL - 0.5
    row = (np.asarray(y, dtype=np.float64) - PC_MIN) / CELL - 0.5 + ROW_OFFSET
    return row, col


def cell_to_phys(row, col):
    """(row, col) -> in-frame physical (x, y). Inverse of phys_to_cell; carries
    -ROW_OFFSET so the GT rasterizer / secondary warp stay consistent with the
    primary sampler."""
    x = (np.asarray(col, dtype=np.float64) + 0.5) * CELL + PC_MIN
    y = (np.asarray(row, dtype=np.float64) - ROW_OFFSET + 0.5) * CELL + PC_MIN
    return x, y


def bilinear_sample(arr, row, col):
    """Bilinear-sample arr[H,W] at float (row, col). Scalars or arrays.

    Returns NaN wherever any of the 4 neighboring cells is out of bounds.
    """
    row = np.asarray(row, dtype=np.float64)
    col = np.asarray(col, dtype=np.float64)
    scalar = (row.ndim == 0)
    row = np.atleast_1d(row)
    col = np.atleast_1d(col)
    H, W = arr.shape
    r0 = np.floor(row).astype(np.int64)
    c0 = np.floor(col).astype(np.int64)
    r1, c1 = r0 + 1, c0 + 1
    valid = (r0 >= 0) & (c0 >= 0) & (r1 < H) & (c1 < W)
    out = np.full(row.shape, np.nan, dtype=np.float64)
    if np.any(valid):
        a = arr.astype(np.float64)
        rv0, cv0 = r0[valid], c0[valid]
        rv1, cv1 = r1[valid], c1[valid]
        fr = row[valid] - rv0
        fc = col[valid] - cv0
        top = a[rv0, cv0] * (1 - fc) + a[rv0, cv1] * fc
        bot = a[rv1, cv0] * (1 - fc) + a[rv1, cv1] * fc
        out[valid] = top * (1 - fr) + bot * fr
    return float(out[0]) if scalar else out


def footprint_mask(boxes):
    """Rasterize BEV box footprints (rotated rectangles) onto the GRID.

    boxes: (N, >=7) as [x, y, z, dx, dy, dz, yaw]. Cell-center membership test.
    Approximate (used by the secondary metric and the frame-check, neither of
    which needs corner-exact rasterization).
    """
    mask = np.zeros((GRID, GRID), dtype=bool)
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.shape[0] == 0:
        return mask
    rr, cc = np.meshgrid(np.arange(GRID), np.arange(GRID), indexing='ij')
    cx, cy = cell_to_phys(rr.astype(np.float64), cc.astype(np.float64))
    for b in boxes:
        x, y, dx, dy, yaw = b[0], b[1], b[3], b[4], b[6]
        cos, sin = math.cos(yaw), math.sin(yaw)
        px, py = cx - x, cy - y
        lx = px * cos + py * sin
        ly = -px * sin + py * cos
        mask |= (np.abs(lx) <= dx / 2.0) & (np.abs(ly) <= dy / 2.0)
    return mask


# --------------------------------------------------------------------------- #
# Loading / scene grouping
# --------------------------------------------------------------------------- #
def load_z_fg(src):
    """Load z_fg and apply the empirically-verified orientation correction.

    The dumped z_fg is rotated 270 deg relative to this analyzer's (row=y,
    col=x) convention. Sweeping all 8 axis orientations and measuring IoU
    between the predicted-foreground mask and the GT-footprint mask on the real
    dumps gave median IoU 0.217 for rot270 (== transpose+fliplr) vs <=0.013 for
    every other orientation (identity, transpose, the flips, the other
    rotations). Root cause: the model's init_bev_mask meshgrid ordering
    (indexing='ij', stack(y_coords, x_coords)) combined with the .flip(1) in
    obtain_bev_mask_gt indexes z_fg rotated 270 deg from this convention.

    Centralized here so the correction is applied identically by the primary
    metric, the secondary metric and the verify-frame gate. ``src`` is an open
    npz / dict (subscriptable) or a path.
    """
    if isinstance(src, (str, bytes)) or hasattr(src, '__fspath__'):
        z = np.load(src, allow_pickle=True)['z_fg']
    else:
        z = src['z_fg']
    return np.rot90(np.asarray(z, dtype=np.float64), k=3)


def load_dumps(dump_dir):
    paths = sorted(glob.glob(osp.join(dump_dir, '*.npz')))
    frames = []
    for p in paths:
        d = np.load(p, allow_pickle=True)
        frames.append(dict(
            z_fg=load_z_fg(d),
            token=str(d['token']),
            scene_token=str(d['scene_token']),
            timestamp=float(d['timestamp']),
            l2e_t=np.asarray(d['lidar2ego_translation'], dtype=np.float64),
            l2e_R=quat_to_rotmat(d['lidar2ego_rotation']),
            e2g_t=np.asarray(d['ego2global_translation'], dtype=np.float64),
            e2g_R=quat_to_rotmat(d['ego2global_rotation']),
            gt_boxes=np.asarray(d['gt_boxes'], dtype=np.float64),
            gt_names=np.asarray(d['gt_names']).astype(str),
            gt_velocity=np.asarray(d['gt_velocity'], dtype=np.float64),
            num_lidar_pts=np.asarray(d['num_lidar_pts']),
        ))
    return frames


def group_scenes(frames, max_scenes=-1):
    by_scene = defaultdict(list)
    for f in frames:
        by_scene[f['scene_token']].append(f)
    scenes = []
    for st, fl in by_scene.items():
        fl = sorted(fl, key=lambda x: x['timestamp'])
        scenes.append((st, fl))
    scenes.sort(key=lambda s: s[1][0]['timestamp'])
    if max_scenes is not None and max_scenes >= 0:
        scenes = scenes[:max_scenes]
    return scenes


# --------------------------------------------------------------------------- #
# PRIMARY metric -- velocity-compensated tracking + center sampling
# --------------------------------------------------------------------------- #
def _greedy_match(pred, nb, names_a, names_b, gate):
    """One-to-one nearest match by ascending residual, same class, within gate."""
    na, nbn = pred.shape[0], nb.shape[0]
    assign = -np.ones(na, dtype=np.int64)
    if na == 0 or nbn == 0:
        return assign
    cand = []
    for i in range(na):
        d = np.linalg.norm(nb - pred[i], axis=1)
        for j in range(nbn):
            if names_a[i] == names_b[j] and d[j] <= gate:
                cand.append((d[j], i, j))
    cand.sort(key=lambda t: t[0])
    used_i, used_j = set(), set()
    for _d, i, j in cand:
        if i in used_i or j in used_j:
            continue
        assign[i] = j
        used_i.add(i)
        used_j.add(j)
    return assign


def _track_scene(frames, velocity_frame='ego'):
    """Return per-frame match arrays linking box i in frame k to k+1 (or -1)."""
    n = len(frames)
    gxy, gvel = [], []
    for f in frames:
        boxes = f['gt_boxes']
        if boxes.shape[0] == 0:
            gxy.append(np.zeros((0, 2)))
            gvel.append(np.zeros((0, 2)))
            continue
        gxy.append(ego_xy_to_global(boxes[:, :2], f['l2e_t'], f['l2e_R'],
                                    f['e2g_t'], f['e2g_R']))
        vel = np.nan_to_num(f['gt_velocity'], nan=0.0)
        if velocity_frame == 'global':
            gvel.append(vel)
        else:
            gvel.append(ego_vec_to_global(vel, f['l2e_R'], f['e2g_R']))

    nxt = [None] * n
    for k in range(n - 1):
        dt = (frames[k + 1]['timestamp'] - frames[k]['timestamp']) * 1e-6
        if dt <= 0:
            warnings.warn(
                f'non-positive dt ({dt:.6g}s) between consecutive frames; '
                'skipping matches for this pair')
            nxt[k] = -np.ones(gxy[k].shape[0], dtype=np.int64)
            continue
        pred = gxy[k] + gvel[k] * dt
        nxt[k] = _greedy_match(pred, gxy[k + 1], frames[k]['gt_names'],
                               frames[k + 1]['gt_names'], MATCH_GATE_M)
    return nxt


def _form_tracks(nxt, box_counts):
    """Chain consecutive-frame matches into tracks: lists of (frame_k, box_i)."""
    n = len(box_counts)
    pointed = [set() for _ in range(n)]
    for k in range(n - 1):
        if nxt[k] is None:
            continue
        for i, j in enumerate(nxt[k]):
            if j >= 0:
                pointed[k + 1].add(int(j))
    tracks = []
    for k in range(n):
        for i in range(box_counts[k]):
            if i in pointed[k]:
                continue  # not a chain start
            chain = [(k, i)]
            ck, ci = k, i
            while ck < n - 1 and nxt[ck] is not None and nxt[ck][ci] >= 0:
                cj = int(nxt[ck][ci])
                chain.append((ck + 1, cj))
                ck, ci = ck + 1, cj
            tracks.append(chain)
    return tracks


def _track_record(chain, frames):
    """Per-track stats + stratification, or None if too few sampled frames."""
    zvals, pts, speeds, name = [], [], [], None
    for (k, i) in chain:
        f = frames[k]
        name = f['gt_names'][i]
        row, col = phys_to_cell(f['gt_boxes'][i, 0], f['gt_boxes'][i, 1])
        z = bilinear_sample(f['z_fg'], row, col)
        if not math.isnan(z):
            zvals.append(z)
        pts.append(float(f['num_lidar_pts'][i]))
        v = np.nan_to_num(f['gt_velocity'][i], nan=0.0)
        speeds.append(float(np.linalg.norm(v)))

    n_frames = len(chain)
    n_sampled = len(zvals)
    if n_sampled < 2:
        return None  # cannot estimate temporal std
    zvals = np.asarray(zvals)
    std = float(np.std(zvals))
    mad = float(np.mean(np.abs(np.diff(zvals)))) if n_sampled >= 2 else 0.0

    # decision-flip metric: threshold the center-z_fg series at FG_LOGIT_THR
    # (raw crossing, no smoothing/hysteresis) and count fg/bg transitions. This
    # is what TFP actually acts on -- an object can have high logit std but
    # stable decisions (logit far from threshold) or vice versa.
    decision = zvals > FG_LOGIT_THR
    fg_fraction = float(np.mean(decision))
    flip_count = int(np.count_nonzero(decision[1:] != decision[:-1]))
    flip_rate = float(flip_count / (n_sampled - 1))  # n_sampled >= 2 here
    is_boundary = bool(flip_count >= 1)

    median_pts = float(np.median(pts))
    mean_speed = float(np.mean(speeds))
    sparsity = 'sparse' if median_pts <= SPARSE_MAX_PTS else 'dense'
    motion = 'static' if mean_speed < STATIC_MAX_SPEED else 'dynamic'
    return dict(
        scene_token=frames[chain[0][0]]['scene_token'],
        gt_name=str(name),
        bucket=f'{sparsity}_{motion}',
        n_frames=n_frames,
        n_sampled=n_sampled,
        std=std,
        mean_abs_delta=mad,
        mean_z=float(np.mean(zvals)),
        fg_fraction=fg_fraction,
        flip_count=flip_count,
        flip_rate=flip_rate,
        is_boundary=is_boundary,
        median_pts=median_pts,
        mean_speed=mean_speed,
    )


def run_primary(scenes, velocity_frame='ego'):
    records = []
    for _st, frames in scenes:
        if len(frames) < MIN_TRACK_FRAMES:
            continue
        nxt = _track_scene(frames, velocity_frame=velocity_frame)
        box_counts = [f['gt_boxes'].shape[0] for f in frames]
        for chain in _form_tracks(nxt, box_counts):
            if len(chain) < MIN_TRACK_FRAMES:
                continue
            rec = _track_record(chain, frames)
            if rec is not None:
                records.append(rec)

    buckets = {}
    for b in BUCKETS:
        recs = [r for r in records if r['bucket'] == b]
        if not recs:
            buckets[b] = dict(
                n_tracks=0, mean=None, median=None, p90=None,
                frac_std_lt_0_30=None, mean_abs_delta=None, mean_center_z=None,
                frac_boundary=None, mean_flip_rate_all=None,
                mean_flip_rate_boundary=None, mean_fg_fraction=None,
                fg_fraction_hist=[0, 0, 0, 0, 0, 0])
            continue
        stds = np.asarray([r['std'] for r in recs])
        mads = np.asarray([r['mean_abs_delta'] for r in recs])
        mzs = np.asarray([r['mean_z'] for r in recs])
        flip_rates = np.asarray([r['flip_rate'] for r in recs])
        fg_fracs = np.asarray([r['fg_fraction'] for r in recs])
        boundary = np.asarray([r['is_boundary'] for r in recs], dtype=bool)
        bnd_rates = flip_rates[boundary]
        hist = np.histogram(fg_fracs, bins=FG_FRACTION_BINS)[0].astype(int)
        buckets[b] = dict(
            n_tracks=len(recs),
            mean=float(np.mean(stds)),
            median=float(np.median(stds)),
            p90=float(np.percentile(stds, 90)),
            frac_std_lt_0_30=float(np.mean(stds < STD_THRESH)),
            mean_abs_delta=float(np.mean(mads)),
            mean_center_z=float(np.mean(mzs)),
            frac_boundary=float(np.mean(boundary)),
            mean_flip_rate_all=float(np.mean(flip_rates)),
            mean_flip_rate_boundary=(float(np.mean(bnd_rates))
                                     if bnd_rates.size else None),
            mean_fg_fraction=float(np.mean(fg_fracs)),
            fg_fraction_hist=hist.tolist(),
        )
    return buckets, records


# --------------------------------------------------------------------------- #
# SECONDARY metric -- full-grid warp + background-baseline subtraction
# --------------------------------------------------------------------------- #
def _dilate(mask, margin):
    out = mask.copy()
    for _ in range(margin):
        d = out.copy()
        d[1:, :] |= out[:-1, :]
        d[:-1, :] |= out[1:, :]
        d[:, 1:] |= out[:, :-1]
        d[:, :-1] |= out[:, 1:]
        out = d
    return out


def _secondary_scene(frames):
    """Per-cell temporal std (warped into frame0 grid) + fg / bg pools."""
    f0 = frames[0]
    rr, cc = np.meshgrid(np.arange(GRID), np.arange(GRID), indexing='ij')
    rr = rr.ravel().astype(np.float64)
    cc = cc.ravel().astype(np.float64)
    x0, y0 = cell_to_phys(rr, cc)
    g = ego_xy_to_global(np.stack([x0, y0], axis=1), f0['l2e_t'], f0['l2e_R'],
                         f0['e2g_t'], f0['e2g_R'])

    stack = np.full((len(frames), rr.size), np.nan)
    for k, f in enumerate(frames):
        rc = global_xy_to_ego_cell(g, f['l2e_t'], f['l2e_R'], f['e2g_t'],
                                   f['e2g_R'])
        stack[k] = bilinear_sample(f['z_fg'], rc[:, 0], rc[:, 1])

    valid_count = np.sum(~np.isnan(stack), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        cell_std = np.nanstd(stack, axis=0)
    cell_std[valid_count < 2] = np.nan
    cell_std = cell_std.reshape(GRID, GRID)
    valid_count = valid_count.reshape(GRID, GRID)

    fg = footprint_mask(f0['gt_boxes'])                       # frame0 anchor
    bg = (~_dilate(fg, margin=2)) & (valid_count >= 2)        # empty region
    fg = fg & (valid_count >= 2)
    return cell_std, fg, bg


def run_secondary(scenes):
    fg_pool, bg_pool = [], []
    for _st, frames in scenes:
        if len(frames) < 2:
            continue
        cell_std, fg, bg = _secondary_scene(frames)
        fg_vals = cell_std[fg]
        bg_vals = cell_std[bg]
        fg_pool.append(fg_vals[~np.isnan(fg_vals)])
        bg_pool.append(bg_vals[~np.isnan(bg_vals)])
    fg_pool = np.concatenate(fg_pool) if fg_pool else np.zeros(0)
    bg_pool = np.concatenate(bg_pool) if bg_pool else np.zeros(0)
    fg_std = float(np.median(fg_pool)) if fg_pool.size else None
    bg_std = float(np.median(bg_pool)) if bg_pool.size else None
    diff = (fg_std - bg_std) if (fg_std is not None and bg_std is not None) \
        else None
    return dict(
        label='SECONDARY (artifact-contaminated, baseline-subtracted)',
        foreground_std_median=fg_std,
        background_std_median=bg_std,
        baseline_subtracted=diff,
        n_foreground_cells=int(fg_pool.size),
        n_background_cells=int(bg_pool.size),
    )


# --------------------------------------------------------------------------- #
# FRAME-VERIFICATION GATE
# --------------------------------------------------------------------------- #
def _shift_mask(mask, dr, dc):
    """Shift boolean mask so element [r,c] moves to [r+dr, c+dc]; zero-filled."""
    out = np.zeros_like(mask)
    H, W = mask.shape
    rs0, rs1 = max(0, -dr), min(H, H - dr)
    cs0, cs1 = max(0, -dc), min(W, W - dc)
    if rs1 > rs0 and cs1 > cs0:
        out[rs0 + dr:rs1 + dr, cs0 + dc:cs1 + dc] = mask[rs0:rs1, cs0:cs1]
    return out


def _mask_iou(a, b):
    union = np.count_nonzero(a | b)
    if union == 0:
        return 0.0
    return np.count_nonzero(a & b) / union


def run_verify(scenes, verify_scenes):
    """Check that gt_boxes (via phys_to_cell) align with predicted foreground."""
    shifts = list(range(-VERIFY_SHIFT, VERIFY_SHIFT + 1))
    per_shift = defaultdict(list)
    n_frames_used = 0
    for sc, (_st, frames) in enumerate(scenes):
        if verify_scenes is not None and verify_scenes >= 0 \
                and sc >= verify_scenes:
            break
        for f in frames:
            gt = footprint_mask(f['gt_boxes'])
            if gt.sum() == 0:
                continue
            pred = f['z_fg'] > FG_LOGIT_THR
            n_frames_used += 1
            for dr in shifts:
                for dc in shifts:
                    per_shift[(dr, dc)].append(
                        _mask_iou(pred, _shift_mask(gt, dr, dc)))

    if n_frames_used == 0:
        return dict(error='no frames with GT boxes found for verification')

    med = {k: float(np.median(v)) for k, v in per_shift.items()}
    best = max(med, key=med.get)
    best_dr, best_dc = best
    best_iou = med[best]
    zero_iou = med[(0, 0)]

    if (best_dr, best_dc) == (0, 0):
        if zero_iou >= VERIFY_IOU_OK:
            verdict = 'OK'
            msg = ('frame alignment OK (best-shift zero, IoU >= '
                   f'{VERIFY_IOU_OK}); primary metric trustworthy')
        else:
            verdict = 'OK_LOOSE'
            msg = ('aligned (best-shift zero) but low absolute IoU '
                   f'({zero_iou:.3f}) = loosely-calibrated foreground '
                   'prediction; fine for the stability metric')
    elif (best_iou - zero_iou) < 0.02:
        verdict = 'OK'
        msg = ('frame alignment OK (best-shift non-zero but IoU gain '
               'negligible -> within quantization noise)')
    else:
        verdict = 'MISALIGNED'
        msg = (f'FRAME MISALIGNED -- best-shift (dr,dc)=({best_dr},{best_dc}) '
               '~ lidar2ego offset; apply this correction before trusting the '
               'primary metric')

    # a few top shifts for context
    top = sorted(med.items(), key=lambda kv: kv[1], reverse=True)[:5]
    return dict(
        n_frames_used=n_frames_used,
        zero_shift_iou_median=zero_iou,
        best_shift=[int(best_dr), int(best_dc)],
        best_shift_iou_median=best_iou,
        top_shifts=[dict(shift=[int(dr), int(dc)], iou=v) for (dr, dc), v in top],
        verdict=verdict,
        message=msg,
    )


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def config_block(velocity_frame):
    return dict(
        GRID=GRID, CELL=CELL, PC_MIN=PC_MIN, ROW_OFFSET=ROW_OFFSET,
        z_fg_orientation='rot270_on_load',
        SPARSE_MAX_PTS=SPARSE_MAX_PTS,
        STATIC_MAX_SPEED=STATIC_MAX_SPEED, MIN_TRACK_FRAMES=MIN_TRACK_FRAMES,
        MATCH_GATE_M=MATCH_GATE_M, STD_THRESH=STD_THRESH,
        FG_LOGIT_THR=FG_LOGIT_THR, velocity_frame=velocity_frame,
    )


def decision_line(buckets):
    lines = []
    ss = buckets.get('sparse_static', {})

    # existing logit-std signal
    if ss.get('n_tracks', 0) > 0 and ss.get('mean') is not None:
        m = ss['mean']
        if m < STD_THRESH:
            lines.append(
                f'[logit-std] sparse_static mean z_fg std = {m:.3f} < '
                f'{STD_THRESH} -> on this signal the foreground logit is '
                'already stable on the hard stratum (case looks WEAK)')
        else:
            lines.append(
                f'[logit-std] sparse_static mean z_fg std = {m:.3f} >= '
                f'{STD_THRESH} -> on this signal the TFP stability case HAS '
                'MERIT')
    else:
        lines.append('[logit-std] sparse_static bucket empty -- cannot '
                     'evaluate')

    # decision-flip signal (the directly-TFP-relevant quantity)
    if ss.get('n_tracks', 0) > 0 and ss.get('frac_boundary') is not None:
        fb = ss['frac_boundary']
        fr = ss['mean_flip_rate_boundary']
        fr_s = f'{fr:.3f}' if fr is not None else 'n/a'
        lines.append(
            f'[decision-flip] sparse_static frac_boundary = {fb:.3f} '
            '(fraction of sparse-static objects whose fg/bg decision flips), '
            f'boundary-track mean flip rate = {fr_s}')
        lines.append(
            '  guidance: high frac_boundary + high boundary flip rate => TFP '
            'has many substantially-flickering targets (strong stability '
            'case); low frac_boundary => most objects are stably decided '
            'regardless of logit noise (weak case -- the logit-std signal '
            'overstates instability)')
    else:
        lines.append('[decision-flip] sparse_static bucket empty -- cannot '
                     'evaluate')

    lines.append('NOTE: final TFP go/no-go also needs the Diagnostic-4 oracle '
                 'mask ceiling.')
    return lines


def print_report(report):
    print('=' * 72)
    print('Diagnostic 1 (P_fg stability) report')
    print('=' * 72)
    print('Config:', json.dumps(report['config']))
    print(f"Scenes analyzed: {report['n_scenes']}  "
          f"frames: {report['n_frames']}  tracks: {report['n_tracks']}")
    print('\nPRIMARY (center-based, velocity-compensated tracking):')
    print(f"  {'bucket':<16}{'n':>5}{'mean':>9}{'median':>9}{'p90':>9}"
          f"{'frac<0.30':>11}{'meanΔ':>9}{'meanZ':>9}")
    for b in ('sparse_static', 'sparse_dynamic', 'dense_static',
              'dense_dynamic'):
        s = report['primary'][b]
        if s['n_tracks'] == 0:
            print(f"  {b:<16}{0:>5}{'-':>9}{'-':>9}{'-':>9}{'-':>11}{'-':>9}"
                  f"{'-':>9}")
        else:
            print(f"  {b:<16}{s['n_tracks']:>5}{s['mean']:>9.3f}"
                  f"{s['median']:>9.3f}{s['p90']:>9.3f}"
                  f"{s['frac_std_lt_0_30']:>11.3f}{s['mean_abs_delta']:>9.3f}"
                  f"{s['mean_center_z']:>9.3f}")

    print('\nPRIMARY -- decision flips (decision = center z_fg > FG_LOGIT_THR):')
    print(f"  {'bucket':<16}{'n':>5}{'frac_bndry':>11}{'flip_all':>10}"
          f"{'flip_bnd':>10}{'fg_frac':>9}")
    for b in BUCKETS:
        s = report['primary'][b]
        if s['n_tracks'] == 0:
            print(f"  {b:<16}{0:>5}{'-':>11}{'-':>10}{'-':>10}{'-':>9}")
            continue
        fb = s['mean_flip_rate_boundary']
        fb_s = f'{fb:.3f}' if fb is not None else '-'
        print(f"  {b:<16}{s['n_tracks']:>5}{s['frac_boundary']:>11.3f}"
              f"{s['mean_flip_rate_all']:>10.3f}{fb_s:>10}"
              f"{s['mean_fg_fraction']:>9.3f}")
    print('  fg_fraction histogram bins '
          '[0-.1)(.1-.3)(.3-.5)(.5-.7)(.7-.9)(.9-1]:')
    for b in BUCKETS:
        s = report['primary'][b]
        print(f"    {b:<16}{s['fg_fraction_hist']}")

    sec = report['secondary']
    print(f"\n{sec['label']}:")
    print(f"  foreground std (median): {sec['foreground_std_median']}")
    print(f"  background std (median): {sec['background_std_median']}")
    print(f"  baseline-subtracted    : {sec['baseline_subtracted']}")
    print('\nDECISION:')
    for line in report['decision']:
        print('  -', line)
    print('=' * 72)


def print_verify(res):
    print('=' * 72)
    print('Diagnostic 1 -- FRAME-VERIFICATION GATE')
    print('=' * 72)
    if 'error' in res:
        print('ERROR:', res['error'])
        return
    print(f"frames used            : {res['n_frames_used']}")
    print(f"zero-shift IoU (median): {res['zero_shift_iou_median']:.3f}")
    print(f"best-shift (dr,dc)     : {tuple(res['best_shift'])}  "
          f"IoU {res['best_shift_iou_median']:.3f}")
    print('top shifts:', ', '.join(
        f"{tuple(s['shift'])}={s['iou']:.3f}" for s in res['top_shifts']))
    print(f"verdict: [{res['verdict']}] {res['message']}")
    print('=' * 72)


# --------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(
        description='Diagnostic 1 analyze: temporal stability of z_fg.')
    parser.add_argument('--dump-dir', default='diag1_dumps',
                        help='directory of per-frame .npz dumps')
    parser.add_argument('--out', default='diag1_report.json',
                        help='output JSON report path')
    parser.add_argument('--max-scenes', type=int, default=-1,
                        help='cap number of scenes analyzed; -1 = all')
    parser.add_argument('--verify-frame', action='store_true',
                        help='run the frame-alignment gate and exit')
    parser.add_argument('--verify-scenes', type=int, default=20,
                        help='scenes used by --verify-frame')
    parser.add_argument('--velocity-frame', choices=['ego', 'global'],
                        default='ego',
                        help='frame gt_velocity is stored in; "global" skips '
                             'the ego->global rotation')
    return parser.parse_args()


def main():
    args = parse_args()
    frames = load_dumps(args.dump_dir)
    if not frames:
        raise SystemExit(f'no .npz dumps found in {args.dump_dir}')
    scenes = group_scenes(frames, args.max_scenes)

    if args.verify_frame:
        res = run_verify(scenes, args.verify_scenes)
        out = osp.splitext(args.out)[0] + '_verify.json'
        with open(out, 'w') as fp:
            json.dump(res, fp, indent=2)
        print_verify(res)
        print(f'(wrote {out})')
        return

    buckets, records = run_primary(scenes, velocity_frame=args.velocity_frame)
    secondary = run_secondary(scenes)
    report = dict(
        config=config_block(args.velocity_frame),
        n_scenes=len(scenes),
        n_frames=sum(len(fl) for _st, fl in scenes),
        n_tracks=len(records),
        primary=buckets,
        per_track=records,
        secondary=secondary,
        decision=decision_line(buckets),
    )
    with open(args.out, 'w') as fp:
        json.dump(report, fp, indent=2)
    print_report(report)
    print(f'(wrote {args.out})')


if __name__ == '__main__':
    main()
