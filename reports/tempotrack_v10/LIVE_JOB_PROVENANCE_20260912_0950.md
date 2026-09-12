# Live job provenance — 2026-09-12 09:50 CST

This receipt was collected before applying the V10.3 online correction. No
process was signalled before this file was written. Paths below are the paths
actually present in `/proc`, not reconstructed from a report.

## Integration repository

- Repository: `/data1/LWR/vranlee/SERVER_ONLY/avis/masa_psmr_v10_unified`
- Branch: `codex/tempotrack-v10-ov-cov-tract-masa`
- HEAD: `420f70e0a93856ce142e39f96bd793f37cba0329`
- Remote branch at collection time: `420f70e0a93856ce142e39f96bd793f37cba0329`
- Dirty before correction:
  - `reports/tempotrack_v10/PROGRESS.md`
  - `reports/tempotrack_v10/reproduction/covtrack.md`
  - untracked `tools/v10_finalize_covtrack_tempo_test.sh`
- No reset, clean, or overwrite was performed.

## Resource snapshot

`free -h` at collection: 125 GiB total, 96 GiB used, 27 GiB available,
no swap. `nvidia-smi` reported ten A100-SXM4-40GB devices. Used/free MiB and
utilization were:

| GPU | UUID | Used | Free | Util |
|---:|---|---:|---:|---:|
| 0 | `GPU-d3a949d3-b3ef-04b0-92c5-594a63857898` | 6972 | 33365 | 27% |
| 1 | `GPU-5de9a1c2-0cc1-4fda-7d25-493f86f52424` | 6900 | 33437 | 0% |
| 2 | `GPU-a4095968-9191-50eb-d56c-ed633b31e2c0` | 6886 | 33451 | 27% |
| 3 | `GPU-f931e51b-55b2-c2c2-6506-b6f986864d54` | 6886 | 33451 | 24% |
| 4 | `GPU-daa9b388-4540-242c-83d7-3261bb232a7c` | 6888 | 33449 | 41% |
| 5 | `GPU-f07a0798-7ddc-ec7c-a27c-d47124bbfe81` | 6886 | 33451 | 57% |
| 6 | `GPU-c2902ddf-62f4-cc2f-5d9f-d33c74555847` | 6800 | 33537 | 26% |
| 7 | `GPU-995c9557-d84e-1a76-f02f-b1ddea34e56b` | 6800 | 33537 | 12% |
| 8 | `GPU-cd127bcd-08c4-fe6a-78ca-fc41123d7119` | 3394 | 36943 | 32% |
| 9 | `GPU-e947d8da-fc7a-4c0b-7d91-e9e67d34f4a6` | 6900 | 33437 | 0% |

## OVTrack Tempo — preserve and continue

All workers used the integration worktree as cwd, official OVTrack source
`/data2/usr_for_deadline/tempotrack_v10_unified/ovtrack_full_source` at
`e188b32eccc049fd425e80b11a3bc45ce88edb31`, config
`configs/research/v10/ovtrack_full_runtime.py` (SHA256
`42d1dcff2e1a7c2316729b259dab3ae737fd1a1bf01994291efa07d8459f55b6`), and
checkpoint `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/ovtrack/saved_models/ovtrack_detpro_prompt.pth`
(SHA256 `76b4605067aacacae87fd8d17207e3fb56f01fa9c0b02b9b2b69d9f5676ace47`).
Runner SHA256:
`bdf056eaea0f649c00a062b1a8637716ea20c22daf682982a6d87ad7861d8f2b`.

- Val parents: `3715` (shard 0, GPU5), `3663` (shard 1, GPU6), `3780`
  (shard 2, GPU7), `3835` (shard 3, GPU8). DataLoader children were
  `7683`, `7777`, `7929`, `7593` respectively.
- Test parents: `9456` (shard 0, GPU0), `9459` (shard 1, GPU2), `9462`
  (shard 2, GPU3), `9465` (shard 3, GPU4). DataLoader children were
  `14063`, `13696`, `13791`, `13941` respectively.
- All observed parents and children were alive and progressing. They were not
  signalled.
- Output root: `/data2/usr_for_deadline/tempotrack_v10_unified/tempo_full/ovtrack/`.

## COVTrack Tempo — preserve and continue

All COV workers used cwd `/data1/LWR/vranlee/SERVER_ONLY/avis/external_ovmot/COVTrack`,
source HEAD `9b0ced5779ee36f5dd73dbe39b5ae5d57abb4b3b` with pre-existing dirty
external changes, config
`/data1/LWR/vranlee/SERVER_ONLY/avis/external_ovmot/COVTrack/configs/uncertainty-ovtrack-teta/ovtrack_r50_ctao_train.py`,
checkpoint
`/data1/LWR/vranlee/SERVER_ONLY/avis/external_ovmot/COVTrack/saved_models/ctao_public_res/ctao_public.pth`
(SHA256 `e4d0b65798844280ea13943e580ce6233ae5d331c4aa1d38eb226c3208dd567c`),
and runner SHA256
`2b9f16f70659a9fdf337848c3892d7ee3caf5a55565151bb7c937bb4541c759e`.
All commands explicitly contained `only_test_categories=True`,
`match_score_thr=0.37`, `memo_frames=50`, `momentum_embed=0.4`,
`confused_features=True`, `vis=False`, `max_per_img=80`, and
`max_fusion_ratio=2.0`; the integration source contains ancestor `420f70e`.

- Single full stream: parent `22665`, child `22689`, GPU9.
- Complete-video shards: `23261` (shard 0, GPU1; child `23798`), `23218`
  (shard 1, GPU5; child `23709`), `23365` (shard 2, GPU6; child `23892`),
  `23353` (shard 3, GPU7; child `23992`).
- Output root: `/data2/usr_for_deadline/tempotrack_v10_unified/tempo_full/covtrack/`.
- Coordinator: `25086`, `tools/v10_finalize_covtrack_tempo_test.sh`.
- All observed COV processes were alive and progressing. They were not
  signalled.

## OVTrack+ Tempo — invalid checkpoint evidence

All six parents used cwd
`/data1/LWR/vranlee/SERVER_ONLY/avis/masa_psmr_v10_unified`, source
`/data2/usr_for_deadline/tempotrack_v10_unified/ovtrack_plus_full_source` at
`f033b314c659995936b1d3becd5baf1deb93e121`, config
`configs/research/v10/ovtrack_plus_full_runtime.py` (SHA256
`579e2acfa9b48997e9fdedceb57e12bfe1c7596c5c6a8ff1461ecd231b444e67`), and
checkpoint
`/data2/usr_for_deadline/tempotrack_v10_unified/checkpoints/ovtrack_plus/ovtrack_clip_distillation.pth`
(SHA256 `36f10026e86d0310c08dac941bea7f68ec4c4d6d3693d38f99bdc3a90e7dc872`).
Runner SHA256:
`4d1c4271559c981c1e0920fd96964d3a38fc4a74b8687241f59869d797c6e149`.

| shard | parent | child | CUDA_VISIBLE_DEVICES | output |
|---:|---:|---:|---:|---|
| 0 | 13007 | 13188 | 9 | `tempo_full/ovtrack_plus/test/shard_0` |
| 1 | 14358 | 14570 | 0 | `tempo_full/ovtrack_plus/test/shard_1` |
| 2 | 14360 | 14821 | 1 | `tempo_full/ovtrack_plus/test/shard_2` |
| 3 | 14362 | 14490 | 2 | `tempo_full/ovtrack_plus/test/shard_3` |
| 4 | 14364 | 14735 | 3 | `tempo_full/ovtrack_plus/test/shard_4` |
| 5 | 14366 | 14612 | 4 | `tempo_full/ovtrack_plus/test/shard_5` |

The retained logs independently show:

```text
load checkpoint from local path: .../ovtrack_clip_distillation.pth
The model and loaded state dict do not match exactly
missing keys in source state_dict:
roi_head.track_head.convs.0.conv.weight, ...,
roi_head.track_head.fcs.0.weight, roi_head.track_head.fcs.0.bias,
roi_head.track_head.fc_embed.weight, roi_head.track_head.fc_embed.bias
```

The same missing track-head list is present in each shard's
`retry_video_id.log` (and in shard 0's retained retry log). This satisfies
the taskbook's two-part invalidity condition: the positional checkpoint is the
upstream training initializer and the runtime log proves missing track-head
parameters. Partial stream line counts at collection were:
`1187, 1229, 1412, 1358, 1010, 1590` for shards 0–5. The lane coordinator
was PID `17125`.

## Checkpoint search

The controlled search found no `epoch_6.pth`, `latest.pth`, `*ovtrack*plus*.pth`,
or `*ovsort*.pth` under the searched `/data1/LWR/vranlee/SERVER_ONLY/avis`
and `/data2/usr_for_deadline` roots. The only matching OVT-B weight found at
the expected experiment root was `ovtrack_clip_distillation.pth`, which is
rejected above. No final OVTrack+ checkpoint is accepted by this receipt.

## Decision recorded after this receipt

OVTrack and COV remain running. The six OVTrack+ parent processes and their
children are eligible for a graceful TERM because their actual checkpoint is
invalid and their logs prove missing track-head parameters. No partial output
is deleted. Until a valid epoch-6 checkpoint is found or trained under safe
RAM, OVTrack+ official reproduction and OVTrack+ memory-only are
`REPRO_BLOCKED_FINAL_CHECKPOINT`.
