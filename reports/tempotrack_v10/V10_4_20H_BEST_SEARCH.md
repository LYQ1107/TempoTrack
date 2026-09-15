# TempoTrack V10.4 Final Report

controller_status: PARTIAL_FAILURE_OR_DEADLINE
controller_pid: 16946
controller_started_at: 2026-09-14T08:14:20Z
controller_finished_at: 2026-09-15T04:14:42Z
search_root: /data2/usr_for_deadline/tempotrack_v10_unified/search/covtrack_v104_best20h_20260914
repository_head: 13a863998d52339306ab1ef09cc20a59ecdc427f
external_source_commit: 9b0ced5779ee36f5dd73dbe39b5ae5d57abb4b3b
contract_gate_sha256: 7fd0167fc57a54a82fbca9c5ff32b1b9aebff62bcded86efff40a94dd704bc1f

All values below are read from the corresponding receipt/result artifact; no old report number is copied as a result.

## Original full COV native baseline

This is the only baseline used for subsequent full-Test metric deltas. The 11,500-frame subset baseline is diagnostic-only.

scope: FULL_TEST
images: 52155
rows: 2353689
annotation: /data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/data/external_annotations/ovtr/tao_test_burst_v1.json
annotation_sha256: f5650b85dba14d3721316121c8141ad0b33e1446583d140fddd1992002b26ec2
summary: /data2/usr_for_deadline/tempotrack_v10_unified/tempo_full/covtrack/test_v10_runtime_gate_sharded/merged/evaluation/COVTrack_V10_Tempo_Test/teta_summary_results.pth
summary_sha256: d8454e184f3bcad01e75d43fd32acaeea2c7c89dbb76d48c5a6c70719b06fdd2
merge_manifest: /data2/usr_for_deadline/tempotrack_v10_unified/tempo_full/covtrack/test_v10_runtime_gate_sharded/merged/merge_manifest.json
merge_manifest_sha256: 739c0b2484e59889a772e3b695bb7c82946c69f4b4bc928bc19132bc7eb9d322
baseline_receipt: /data2/usr_for_deadline/tempotrack_v11_qdic_fulltest_20h_20260914/full_results.json
baseline_receipt_sha256: 9734f50a09855b15360e6d613954e0af300cedbb18cdb99ed0396c70d2e85cab

| split | TETA | LocA | AssocA | ClsA | LocRe | LocPr | AssocRe | AssocPr | ClsRe | ClsPr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 37.166000 | 54.841000 | 41.149000 | 15.507000 | 57.442000 | 78.600000 | 48.168000 | 58.996000 | 25.692000 | 22.008000 |
| base | 38.028515 | 55.181577 | 42.157859 | 16.746170 | 57.827614 | 79.058623 | 48.923338 | 60.413143 | 27.959465 | 23.269283 |
| novel | 28.695906 | 51.500333 | 31.247248 | 3.340153 | 53.655455 | 74.092242 | 40.750485 | 45.080636 | 3.431571 | 9.629545 |

## COV Wave2 subset search (DIAGNOSTIC_ONLY)

These rows are not used for the original-baseline delta and are not full-Test claims.

| source | trial | max_gap | candidate_K | score | margin | Base TETA | Base AssocA | Novel TETA | Novel AssocA | Overall TETA |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reused_subset | s03_m01 | 360 | 8 | 1.869273 | 0.065577 | 37.338759 | 40.661179 | 29.450752 | 33.071161 | 36.610000 |
| new_subset | a2_g120_k8__retry03 | 120 | 8 | 1.869273 | 0.065577 | 37.338759 | 40.661179 | 29.450752 | 33.071161 | 36.610000 |
| new_subset | a2_g240_k8__retry01 | 240 | 8 | 1.869273 | 0.065577 | 37.338759 | 40.661179 | 29.450752 | 33.071161 | 36.610000 |
| new_subset | a2_g360_k16__retry01 | 360 | 16 | 1.869273 | 0.065577 | 37.356144 | 40.667881 | 29.177815 | 32.931264 | 36.600000 |
| new_subset | a2_g120_k16__retry03 | 120 | 16 | 1.869273 | 0.065577 | 37.356144 | 40.667881 | 29.177815 | 32.931264 | 36.600000 |
| new_subset | a2_g240_k16__retry01 | 240 | 16 | 1.869273 | 0.065577 | 37.356144 | 40.667881 | 29.177815 | 32.931264 | 36.600000 |
| reused_subset | s03_m00 | 360 | 8 | 1.869273 | 0.000000 | 37.288868 | 40.597730 | 29.706691 | 33.784879 | 36.588000 |
| reused_subset | s02_m00 | 360 | 8 | 1.367089 | 0.000000 | 37.321238 | 40.344604 | 29.281885 | 32.289185 | 36.578000 |
| new_subset | a2_g240_k4__retry01 | 240 | 4 | 1.869273 | 0.065577 | 37.260884 | 40.464210 | 29.363509 | 32.740900 | 36.531000 |
| new_subset | a2_g120_k4__retry03 | 120 | 4 | 1.869273 | 0.065577 | 37.260884 | 40.464210 | 29.363509 | 32.740900 | 36.531000 |
| new_subset | a2_g360_k4__retry01 | 360 | 4 | 1.869273 | 0.065577 | 37.260884 | 40.464210 | 29.363509 | 32.740900 | 36.531000 |
| reused_subset | s03_m02 | 360 | 8 | 1.869273 | 0.131154 | 37.265755 | 40.415701 | 29.156200 | 32.829258 | 36.516000 |
| reused_subset | q1_score_p25_margin0 | 360 | 8 | 2.371458 | 0.000000 | 37.215285 | 40.723923 | 29.345648 | 33.454855 | 36.488000 |
| reused_subset | s02_m01 | 360 | 8 | 1.367089 | 0.065577 | 37.213110 | 40.167168 | 29.015188 | 31.167964 | 36.455000 |
| resumed_current_root | a1_smid_low_mmid_low | 360 | 8 | 1.869273 | 0.397206 | 37.224322 | 40.407430 | 28.848642 | 31.890209 | 36.450000 |
| resumed_current_root | a1_s25_m05 | 360 | 8 | 2.371458 | 0.131154 | 37.172643 | 40.564015 | 28.917976 | 32.848836 | 36.410000 |
| reused_subset | s01_m00 | 360 | 8 | 0.683544 | 0.000000 | 37.124050 | 39.646281 | 29.368727 | 32.382185 | 36.407000 |
| reused_subset | s02_m02 | 360 | 8 | 1.367089 | 0.131154 | 37.170830 | 40.046846 | 28.715909 | 30.966709 | 36.389000 |
| reused_subset | s02_m03 | 360 | 8 | 1.367089 | 0.397206 | 37.104811 | 40.026902 | 29.142239 | 32.289458 | 36.369000 |
| reused_subset | s00_m03 | 360 | 8 | 0.000000 | 0.397206 | 37.057356 | 39.813552 | 28.843000 | 31.831039 | 36.298000 |
| reused_subset | s01_m03 | 360 | 8 | 0.683544 | 0.397206 | 37.058510 | 39.814452 | 28.832970 | 31.790345 | 36.298000 |
| reused_subset | s00_m00 | 360 | 8 | 0.000000 | 0.000000 | 37.001928 | 39.583375 | 29.359818 | 32.355479 | 36.296000 |
| reused_subset | s02_m05 | 360 | 8 | 1.367089 | 1.114677 | 37.018724 | 40.475029 | 28.756773 | 31.858042 | 36.255000 |
| reused_subset | s00_m05 | 360 | 8 | 0.000000 | 1.114677 | 37.012403 | 40.456769 | 28.756773 | 31.858042 | 36.249000 |
| reused_subset | s01_m05 | 360 | 8 | 0.683544 | 1.114677 | 37.012403 | 40.456769 | 28.756773 | 31.858042 | 36.249000 |
| reused_subset | s00_m01 | 360 | 8 | 0.000000 | 0.065577 | 36.982809 | 39.545502 | 28.845415 | 30.561500 | 36.231000 |
| reused_subset | s01_m01 | 360 | 8 | 0.683544 | 0.065577 | 36.981556 | 39.481156 | 28.857930 | 30.531227 | 36.231000 |
| resumed_current_root | a1_s25_m25 | 360 | 8 | 2.371458 | 0.663257 | 36.975308 | 40.450708 | 28.863712 | 32.503445 | 36.226000 |
| reused_subset | s02_m04 | 360 | 8 | 1.367089 | 0.663257 | 36.938802 | 40.126161 | 29.130076 | 32.743885 | 36.217000 |
| reused_subset | s01_m02 | 360 | 8 | 0.683544 | 0.131154 | 36.920071 | 39.509153 | 28.910748 | 30.747782 | 36.180000 |
| reused_subset | s00_m02 | 360 | 8 | 0.000000 | 0.131154 | 36.876935 | 39.388452 | 28.936688 | 30.771782 | 36.143000 |
| reused_subset | s00_m08 | 360 | 8 | 0.000000 | 2.349147 | 36.882835 | 40.262058 | 28.843542 | 31.991173 | 36.140000 |
| reused_subset | s01_m08 | 360 | 8 | 0.683544 | 2.349147 | 36.882835 | 40.262058 | 28.843542 | 31.991173 | 36.140000 |
| reused_subset | s02_m08 | 360 | 8 | 1.367089 | 2.349147 | 36.882835 | 40.262058 | 28.843542 | 31.991173 | 36.140000 |
| reused_subset | q1_score_p50_margin0 | 360 | 8 | 3.582125 | 0.000000 | 36.878537 | 40.223850 | 28.833512 | 32.057082 | 36.135000 |
| reused_subset | s00_m07 | 360 | 8 | 0.000000 | 1.957622 | 36.878403 | 40.255962 | 28.840576 | 31.987112 | 36.135000 |
| reused_subset | s01_m07 | 360 | 8 | 0.683544 | 1.957622 | 36.878403 | 40.255962 | 28.840576 | 31.987112 | 36.135000 |
| reused_subset | s02_m07 | 360 | 8 | 1.367089 | 1.957622 | 36.878403 | 40.255962 | 28.840576 | 31.987112 | 36.135000 |
| reused_subset | s00_m04 | 360 | 8 | 0.000000 | 0.663257 | 36.829889 | 39.996877 | 29.136773 | 32.764885 | 36.119000 |
| reused_subset | s01_m04 | 360 | 8 | 0.683544 | 0.663257 | 36.829914 | 39.996873 | 29.136773 | 32.764885 | 36.119000 |
| resumed_current_root | a1_smid_high_mmid_high | 360 | 8 | 2.976791 | 1.114677 | 36.851460 | 40.285076 | 28.838797 | 31.968415 | 36.111000 |
| resumed_current_root | a1_s50_m25 | 360 | 8 | 3.582125 | 0.663257 | 36.848725 | 40.172033 | 28.748179 | 31.976279 | 36.100000 |
| reused_subset | s00_m06 | 360 | 8 | 0.000000 | 1.566098 | 36.831513 | 40.201935 | 28.831712 | 31.948597 | 36.092000 |
| reused_subset | s01_m06 | 360 | 8 | 0.683544 | 1.566098 | 36.831513 | 40.201935 | 28.831712 | 31.948597 | 36.092000 |
| reused_subset | s02_m06 | 360 | 8 | 1.367089 | 1.566098 | 36.831510 | 40.201932 | 28.831712 | 31.948597 | 36.092000 |
| resumed_current_root | a1_s25_m50 | 360 | 8 | 2.371458 | 1.566098 | 36.784706 | 40.074555 | 28.834621 | 31.969233 | 36.050000 |

## COV Wave2 full Test

| source | spec | Base TETA | Base LocA | Base AssocA | Novel TETA | Novel LocA | Novel AssocA | Overall TETA | prediction SHA256 | summary SHA256 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| existing_full_champion | `{"candidate_top_k":8,"margin_threshold":0.0,"max_gap":360,"score_threshold":2.371457517147064}` | 38.196760 | 54.716549 | 42.696631 | 29.320436 | 51.457667 | 33.454855 | 37.376000 | `16c8c85082406a29a7ef694df33b10377ddee2fadd48e9f6962512c397693c12` | `c259ca4be332136d785eda6db43c09b0158e63bc37f0965ff22f5c48d2d27adc` |
| recovered_full | `{"candidate_top_k":8,"margin_threshold":0.0655771791934967,"max_gap":360,"score_threshold":1.367088943719864}` | 38.227673 | 55.239390 | 42.309218 | 28.987370 | 51.549091 | 31.167964 | 37.374000 | `728e8eb4e6903c98ca5c867df1b1b6eaa558cddffd1c1cc80324942de5b58fd1` | `03a8ca999e9a5d49f1d6e61cd2b1346d7bab5b4506d4145ae44dfa9c9123b62e` |
| new_full | `{"candidate_top_k":8,"margin_threshold":0.0,"max_gap":360,"score_threshold":1.618181087076664}` | 38.223052 | 55.091078 | 42.487295 | 29.737142 | 51.181818 | 33.810785 | 37.439000 | `b660474ecae57b7cd6003ad9ccf33269adbcdf1b0d5d9b3e3038d6ae821d35cf` | `265f61aa8c1a806678d3b54551348f72dbf84da7a6011dc53030cfa9668dc258` |
| new_full | `{"candidate_top_k":8,"margin_threshold":0.0,"max_gap":360,"score_threshold":1.367088943719864}` | 38.397700 | 55.211359 | 42.533397 | 29.243703 | 51.197000 | 32.289185 | 37.552000 | `90b3618219f556be5adb1a0e6f4efe62eabb8af8605714c33c82fc099583c167` | `28b17f7cab571d5a3abedba56ed127a6f7074d44248df0fd50d0c11421d7251a` |
| new_full | `{"candidate_top_k":8,"margin_threshold":0.0655771791934967,"max_gap":360,"score_threshold":1.869273230433464}` | 38.255602 | 54.999661 | 42.657384 | 29.417782 | 51.306667 | 33.071161 | 37.439000 | `cf4bc24f57f78e94c964ad9c374bf2b14beea55e7c32985668949e3ae25c8859` | `9b23f218fd8b22b5e78bf0dd375cb9b67efc77c51000367ec2c243be0bc4e512` |

## Full-Test deltas vs original full COV native baseline

Positive/negative deltas below are candidate minus the complete original baseline above; no subset value is involved.

| source/spec | split | TETA | LocA | AssocA | ClsA | LocRe | LocPr | AssocRe | AssocPr | ClsRe | ClsPr |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `{"candidate_top_k":8,"margin_threshold":0.0,"max_gap":360,"score_threshold":2.371457517147064}` | overall | +0.210000 | -0.426000 | +0.693000 | +0.364000 | -0.515000 | +0.240000 | -1.684000 | +7.214000 | +0.949000 | +0.626000 |
| `{"candidate_top_k":8,"margin_threshold":0.0,"max_gap":360,"score_threshold":2.371457517147064}` | base | +0.168246 | -0.465028 | +0.538772 | +0.430937 | -0.572185 | +0.252012 | -1.735655 | +7.067839 | +1.050867 | +1.062444 |
| `{"candidate_top_k":8,"margin_threshold":0.0,"max_gap":360,"score_threshold":2.371457517147064}` | novel | +0.624530 | -0.042667 | +2.207606 | -0.291444 | +0.044879 | +0.127242 | -1.177273 | +8.647879 | -0.053984 | -3.658848 |
| `{"candidate_top_k":8,"margin_threshold":0.0655771791934967,"max_gap":360,"score_threshold":1.367088943719864}` | overall | +0.208000 | +0.057000 | +0.130000 | +0.436000 | -0.042000 | +0.345000 | -1.333000 | +3.359000 | +0.941000 | +0.438000 |
| `{"candidate_top_k":8,"margin_threshold":0.0655771791934967,"max_gap":360,"score_threshold":1.367088943719864}` | base | +0.199158 | +0.057813 | +0.151359 | +0.388262 | -0.061184 | +0.366738 | -1.235694 | +3.396089 | +0.919170 | +0.623805 |
| `{"candidate_top_k":8,"margin_threshold":0.0655771791934967,"max_gap":360,"score_threshold":1.367088943719864}` | novel | +0.291464 | +0.048758 | -0.079285 | +0.904964 | +0.145424 | +0.137091 | -2.282424 | +2.992170 | +1.151940 | -1.387485 |
| `{"candidate_top_k":8,"margin_threshold":0.0,"max_gap":360,"score_threshold":1.618181087076664}` | overall | +0.273000 | -0.111000 | +0.536000 | +0.394000 | -0.175000 | +0.257000 | -1.138000 | +4.232000 | +0.896000 | +0.621000 |
| `{"candidate_top_k":8,"margin_threshold":0.0,"max_gap":360,"score_threshold":1.618181087076664}` | base | +0.194538 | -0.090499 | +0.329436 | +0.344626 | -0.177558 | +0.301799 | -1.212951 | +4.057419 | +0.866626 | +0.844337 |
| `{"candidate_top_k":8,"margin_threshold":0.0,"max_gap":360,"score_threshold":1.618181087076664}` | novel | +1.041236 | -0.318515 | +2.563536 | +0.878691 | -0.152727 | -0.173818 | -0.397970 | +5.946261 | +1.181061 | -1.579303 |
| `{"candidate_top_k":8,"margin_threshold":0.0,"max_gap":360,"score_threshold":1.367088943719864}` | overall | +0.386000 | -0.001000 | +0.437000 | +0.721000 | -0.075000 | +0.255000 | -0.901000 | +3.436000 | +1.608000 | +0.747000 |
| `{"candidate_top_k":8,"margin_threshold":0.0,"max_gap":360,"score_threshold":1.367088943719864}` | base | +0.369185 | +0.029782 | +0.375538 | +0.702203 | -0.074786 | +0.318142 | -0.836793 | +3.341798 | +1.651153 | +0.974669 |
| `{"candidate_top_k":8,"margin_threshold":0.0,"max_gap":360,"score_threshold":1.367088943719864}` | novel | +0.547797 | -0.303333 | +1.041936 | +0.904782 | -0.073424 | -0.362394 | -1.532182 | +4.358806 | +1.183182 | -1.498091 |
| `{"candidate_top_k":8,"margin_threshold":0.0655771791934967,"max_gap":360,"score_threshold":1.869273230433464}` | overall | +0.273000 | -0.183000 | +0.622000 | +0.379000 | -0.269000 | +0.278000 | -1.447000 | +5.606000 | +0.790000 | +0.458000 |
| `{"candidate_top_k":8,"margin_threshold":0.0655771791934967,"max_gap":360,"score_threshold":1.869273230433464}` | base | +0.227087 | -0.181916 | +0.499525 | +0.363484 | -0.296357 | +0.333565 | -1.486188 | +5.514663 | +0.789567 | +0.692917 |
| `{"candidate_top_k":8,"margin_threshold":0.0655771791934967,"max_gap":360,"score_threshold":1.869273230433464}` | novel | +0.721876 | -0.193667 | +1.823912 | +0.535418 | -0.003727 | -0.267303 | -1.061091 | +6.502970 | +0.793515 | -1.854636 |

## Preserved downstream Test/Val artifacts

| method/split | status | Base TETA | Base LocA | Base AssocA | Base ClsA | Novel TETA | Novel LocA | Novel AssocA | Novel ClsA | result SHA256 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| cov_val | PASS | 38.279097 | 56.818201 | 42.529887 | 15.489245 | 33.277040 | 56.630200 | 40.658206 | 2.542803 | `9c1507b1c255cb2a8824408dc1d83cfe040e65c7877d5fe07dcfe6fe8ec5afc0` |
| cov_test | PASS | 38.196760 | 54.716549 | 42.696631 | 17.177107 | 29.320436 | 51.457667 | 33.454855 | 3.048708 | `e38c333c2ac3e67d33b4936798889b1622a2af5b04188f285ec593cf0ea2336d` |
| ov_val | PASS | 22.140988 | 42.592492 | 8.303471 | 15.526953 | 16.874246 | 39.909714 | 9.147557 | 1.565483 | `c0fdba9aad5ddd6eb72069baab2f72a093b59fa41a918e5cc969fcb1f37201f7` |
| ov_test | PASS | 19.707512 | 39.460629 | 7.070512 | 12.591251 | 14.566048 | 34.852100 | 7.330388 | 1.515587 | `e267132999f0007b2de0f0ddfd811a71a44fc42bc5618f8042c2ef9b33bcc63b` |
| masa_val_native | PASS | 34.280726 | 56.226712 | 36.083245 | 10.532157 | 30.611551 | 55.679486 | 33.655143 | 2.500099 | `72a2e2950d86b9b53193a4a80f69a8ee0dfc92a45fcd749cbb9e3bcb82f5dda0` |
| masa_test_native | PASS | 33.181984 | 54.296173 | 36.291408 | 8.958369 | 26.995912 | 49.635061 | 27.886224 | 3.466488 | `3e42a01175ce4877358ab27d8793709856477a28a068eb551d46edf3bc22b861` |
| masa_val_tempo | PASS | 33.184813 | 55.048079 | 34.441381 | 10.064971 | 30.015034 | 54.947114 | 32.772220 | 2.325971 | `b7beab1506b79b8bd77efa903aba503095eb49a17d6e46ebf1434d0bcf0bb4ba` |
| masa_test_tempo | PASS | 31.923885 | 53.236806 | 33.484466 | 9.050387 | 26.057533 | 48.825394 | 26.382461 | 2.964787 | `89155592767d47ab21706bea00d83b938ad6934a0dd735add172049f60cf7f76` |

## Verification

subset_results_sha256: 1f8c763a727adec34084898ee87c9f144fb148853bd9ef57158746e9d6ec0a6c
full_results_sha256: cc04d520c0f17e285d4f3300a7ed0f4e22a891215c02083692658c97068efdfc
Each PASS full row was checked for prediction and summary file existence and matching SHA256. Downstream rows were checked for status=PASS and matching result-file SHA256.
No detector/native feature export was rerun by this controller; existing artifacts were read-only inputs.

## Resource receipt

{"compute_apps": {"0": [{"name": "/home/lwr/anaconda3/envs/ovtr/bin/python", "pid": 36572, "used_mib": "3408"}, {"name": "/home/lwr/anaconda3/envs/ovtr/bin/python", "pid": 8604, "used_mib": "3212"}], "1": [{"name": "/home/lwr/anaconda3/envs/ovtr/bin/python", "pid": 36574, "used_mib": "3408"}], "2": [{"name": "/home/lwr/anaconda3/envs/ovtr/bin/python", "pid": 36570, "used_mib": "3212"}, {"name": "/home/lwr/anaconda3/envs/ovtr/bin/python", "pid": 10189, "used_mib": "3406"}], "3": [{"name": "/home/lwr/anaconda3/envs/ovtr/bin/python", "pid": 36592, "used_mib": "3408"}, {"name": "/home/lwr/anaconda3/envs/ovtr/bin/python", "pid": 6332, "used_mib": "3408"}], "4": [{"name": "/home/lwr/anaconda3/envs/ovtr/bin/python", "pid": 36594, "used_mib": "3408"}], "5": [{"name": "/home/lwr/anaconda3/envs/ovtr/bin/python", "pid": 36595, "used_mib": "3408"}], "6": [{"name": "/home/lwr/anaconda3/envs/ovtr/bin/python", "pid": 36577, "used_mib": "3408"}], "7": [{"name": "/home/lwr/anaconda3/envs/ovtr/bin/python", "pid": 36585, "used_mib": "3408"}], "8": [{"name": "/home/lwr/anaconda3/envs/ovtr/bin/python", "pid": 36602, "used_mib": "3408"}], "9": [{"name": "/home/lwr/anaconda3/envs/ovtr/bin/python", "pid": 36588, "used_mib": "3214"}]}, "gpu_rows": [{"free_mib": 36927, "index": "4", "used_mib": 3410, "util": 0, "uuid": "GPU-daa9b388-4540-242c-83d7-3261bb232a7c"}, {"free_mib": 36927, "index": "5", "used_mib": 3410, "util": 0, "uuid": "GPU-f07a0798-7ddc-ec7c-a27c-d47124bbfe81"}, {"free_mib": 36927, "index": "6", "used_mib": 3410, "util": 0, "uuid": "GPU-c2902ddf-62f4-cc2f-5d9f-d33c74555847"}, {"free_mib": 36927, "index": "7", "used_mib": 3410, "util": 0, "uuid": "GPU-995c9557-d84e-1a76-f02f-b1ddea34e56b"}, {"free_mib": 36927, "index": "8", "used_mib": 3410, "util": 18, "uuid": "GPU-cd127bcd-08c4-fe6a-78ca-fc41123d7119"}, {"free_mib": 37121, "index": "9", "used_mib": 3216, "util": 0, "uuid": "GPU-e947d8da-fc7a-4c0b-7d91-e9e67d34f4a6"}], "leased": ["0", "1", "2", "3"], "mem_available_gib": 56.24160385131836, "safe_gpu_indices": ["4", "5", "6", "7", "8", "9"], "shared_gpu_policy": "free_vram_at_least_10000_mib", "timestamp": "2026-09-15T04:14:41Z"}

## Incomplete work (transparent status)

This report does not claim that every planned search shard completed. The controller ended in a terminal partial/deadline state; incomplete and failed attempts remain recorded in the state receipt and were not converted into metric rows.
controller_terminal_status: PARTIAL_FAILURE_OR_DEADLINE
incomplete_jobs: ["A2:a2_g120_k32:STOPPED_DEADLINE", "A2:a2_g240_k32:STOPPED_DEADLINE", "A2:a2_g360_k32:STOPPED_DEADLINE", "A3:a3_gap60:STOPPED_DEADLINE", "A3:a3_score_lower:STOPPED_DEADLINE", "A3:a3_score_upper:STOPPED_DEADLINE", "A3:a3_margin_upper:STOPPED_DEADLINE"]
incomplete_full: []
