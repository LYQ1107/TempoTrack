# TempoTrack V8 resource and execution log

This log records observed resources and ownership decisions for the V8 run. It
does not treat an external process as a veto, and no external process was
signalled, stopped, reniced, or modified.

## Repository and environments

- V8 worktree: `/data1/LWR/vranlee/SERVER_ONLY/avis/masa_psmr_v8`
- Branch: `codex/tempotrack-psmr-v8-crossbaseline-test`
- Source snapshot at the start of execution: `82606c9`; V8 source was pushed
  before experiments began. Final commit is recorded in this report after the
  supporting artifacts are committed.
- MASA environment: `/home/lwr/anaconda3/envs/masaenv/bin/python`, with
  `LD_PRELOAD=/home/lwr/anaconda3/envs/masaenv/lib/libsqlite3.so.3.52.0`.
- VOV environment: `/home/lwr/anaconda3/envs/ovtr/bin/python` (Python 3.8.20,
  torch 1.10.1+cu113, mmcv 1.3.17, mmdet 2.23.0). TETA source and the local
  NumPy compatibility adapter were supplied through `PYTHONPATH`; masaenv was
  not modified.

## Input and code hashes

- MASA Test annotation: `0892a2ec8591f41912c5aa2562462162875b15cc6686ce7e508e1d989192c37e`.
- MASA Val annotation: `6372076018b7a2158106e90e6d5de8800fa50263b3af25c1a907ce49e5a5123a`.
- External Val annotation: `1347fdc4e3eb6880f957e81014fa505080b55d895f27afcc3d6343cff4dc6fe7`.
- External Test annotation: `f5650b85dba14d3721316121c8141ad0b33e1446583d140fddd1992002b26ec2`.
- Annotation manifest: `reports/tempotrack_v8/TEST_ANNOTATION_MANIFEST.json`, SHA256
  `d3ce066ff6967577ed9ceebc5ee911916a068e4b5683b00eb70beeb610a3d5d6`.
- `configs/research/psmr_v8.yaml`: `f382fb3e14a44e91d4c2d87f936b59a68dfb49ea2d7f7e9839ae44a3ac57c3e4`.
- `tempotrack_research/cli.py`: `364f8faf6856f09333685209e2694642ae71fa10d35a4dff9bae8263ed6f7d71`.
- `tempotrack_research/orchestration/psmr_v7.py`: `c810e7caef9e0e1f660d611c6992f5ccd0c8cf357e5d92c6c7fcac3f98c329d1`.
- `tempotrack_research/orchestration/v8_crossbaseline.py`:
  `d50ca4d27143efe03b7db1cfaa161b4f0401b525d209d0254d95059fcbbefee1`.

## Resource snapshots

At 2026-09-10 06:45 CST, `nvidia-smi` reported 10 x 40,960 MiB devices:

| device | UUID | used/free MiB | utilization | V8 ownership |
|---|---|---:|---:|---|
| 0 | `GPU-d3a949d3-b3ef-04b0-92c5-594a63857898` | 5,660 / 34,677 | 81% | external VOV Test worker sharing |
| 1 | `GPU-5de9a1c2-0cc1-4fda-7d25-493f86f52424` | 5,632 / 34,705 | 67% | external VOV Test worker sharing |
| 2 | `GPU-a4095968-9191-50eb-d56c-ed633b31e2c0` | 5,632 / 34,705 | 71% | external VOV Test worker sharing |
| 3 | `GPU-f931e51b-55b2-c2c2-6506-b6f986864d54` | 5,610 / 34,727 | 83% | external VOV Test worker sharing |
| 4 | `GPU-daa9b388-4540-242c-83d7-3261bb232a7c` | 2,580 / 37,757 | 7% | unrelated LocateMOT process |
| 5 | `GPU-f07a0798-7ddc-ec7c-a27c-d47124bbfe81` | 0 / 40,337 | 0% | available |
| 6 | `GPU-c2902ddf-62f4-cc2f-5d9f-d33c74555847` | 0 / 40,337 | 0% | available |
| 7 | `GPU-995c9557-d84e-1a76-f02f-b1ddea34e56b` | 0 / 40,337 | 0% | available |
| 8 | `GPU-cd127bcd-08c4-fe6a-78ca-fc41123d7119` | 0 / 40,337 | 0% | available |
| 9 | `GPU-e947d8da-fc7a-4c0b-7d91-e9e67d34f4a6` | 4,696 / 35,641 | 0% | VOV native PSMR Val replay |

The same snapshot reported 125 GiB RAM, 67 GiB available, and no swap. The
VOV Test recorder was allowed to share devices 0--3 because its measured model
footprint was about 2.3 GiB per device in addition to the pre-existing load;
the project never claimed exclusive ownership of those cards.

## Live jobs observed during the run

At 06:46 CST, project-owned jobs were:

- PID 26884: VOV native C10-B1 Val replay, GPU9, alive and CPU-active.
- PID 37076 with workers 37163, 37164, 37166, 37167: corrected VOV native
  Test recorder, four workers on devices 0--3, using Test class operating
  point (`torch.Size([357, 512])`).

Pre-existing processes included OCD phase jobs on devices 0--3 and LocateMOT
on device 4. They were left untouched. The first VOV Test recorder is retained
as a failed/mismatched artifact: it used 296 validation classes and produced
1,653,382 rows, whereas the official Test baseline used 357 classes and
2,918,121 rows. The corrected recorder is a separate output root and is the
only one eligible for VOV Test native replay.

## Failure evidence retained

- The unbounded COV Val DDP attempt hit a kernel OOM during result collection;
  see `reports/tempotrack_v8/cov_val_ddp.log`.
- The first VOV native Test alignment failed with
  `native/official observation join mismatch: missing=1653382, extra=2918121`.
  This was traced to the validation-only class switch, not silently discarded.
- Bounded streaming and separate per-rank recorder outputs were used after
  those failures. The failure artifacts remain in place.

## Final resource and ownership snapshot (2026-09-10 10:37 CST)

- All V8-owned replay/evaluation processes have exited; the final VOV Test
  `evaluate-v6` wrote `evaluation.json` with `status=COMPLETED`.  A final
  `pgrep -af 'tempotrack|eval_ovmot_teta'` found no remaining project worker.
- GPUs 0--3, 5--9 were at `0 MiB` V8 usage and `40,337 MiB` free.  GPU4 had
  `1,957 MiB` used by an unrelated process and was not touched.  No external
  process was killed, stopped, reniced, or reset.
- RAM: 125 GiB total, 115 GiB available; swap remained disabled.  Disk free
  space was approximately 22 GiB on `/data1` and 583 GiB on `/data2`; large
  cross-baseline artifacts remain on the `/data2` relocation backing the
  output symlink.
- The final VOV Test PSMR official association-only summary is
  `outputs/tempotrack_v8/crossbaseline/vov_test_c10_b1_official_evaluation/association_only/vov_test_c10_b1/teta_summary_results.pth`
  (SHA256
  `e01cc80f4ec9d0fc78adeb4530b87207dff9b445fb024986efedf2a0162051b6`).
