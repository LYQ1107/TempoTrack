# COVTrack reproduction and native-feature gate

## Official baseline provenance

- source commit: `9b0ced5779ee36f5dd73dbe39b5ae5d57abb4b3b`
- config SHA256: `282468d93c21b153b755047398b2fe8e95a8d67c003175309e337f9ed5bb600a`
- public checkpoint SHA256: `e4d0b65798844280ea13943e580ce6233ae5d331c4aa1d38eb226c3208dd567c`
- external Val annotation SHA256: `1347fdc4e3eb6880f957e81014fa505080b55d895f27afcc3d6343cff4dc6fe7`
- external Test annotation SHA256: `f5650b85dba14d3721316121c8141ad0b33e1446583d140fddd1992002b26ec2`

## Official TETA baseline

COVTrack's published table has no ClsA; only TETA/LocA/AssocA are reported.

| split | Base | Novel | prediction SHA256 |
|---|---|---|---|
| Val | `39.554/57.155/41.960` | `34.205/58.173/40.936` | `a7ab630543132923839c5da70a0738ff7ef0d39b8b4c4668d5151cecce6abd3b` |
| Test | `37.779/54.598/42.107` | `28.686/50.966/31.960` | `0e51bf048d36f6d9848bcc0d93112c185b7ce0327dc733de91463e73ef2ed876` |

Published-minus-reproduced deltas (reproduced minus published) are Val Base
`-0.046/-0.145/-0.040`, Val Novel `-0.095/-0.027/-0.364`; Test Base
`-0.121/+0.098/+0.007`, Test Novel `-0.214/+0.066/-0.640`.

## Native C9 gate

The native cache has 1,529,908 rows, 988 videos, dimension 256; cache content
SHA256 `f24c747904d329934d49bdff8584fc866d39ee83b7447d64338a3bd301e0a9e5`.
The external analysis used 1,282 pairs (651 positive/631 hard negative), with
MeanCos separation `0.31632668` (AUC `0.8593995`), Top1 `0.3501534` (AUC
`0.8912291`), Top3 `0.3381367` (AUC `0.8778400`), Top5 `0.3319346` (AUC
`0.8727668`), and PaperEMD `0.03881445` (AUC `0.5421231`).

C9 calibration met the precision floor: B=1 threshold `0.83424237`, margin
`0.01`, accepted 256, precision `0.9609375`; B=4 threshold `0.73401513`,
margin `0.1`, accepted 370, precision `0.954054`.

The raw native prediction had 380,329 category mismatches against the official
frontend. The corrected aligned object is
`f23e1cc765b990118980f5aaf79f7f91652754668bf37c71b4bfa1f2a332418`; it has
1,529,908 rows, immutable boxes/scores/UIDs, and retains native association
scores. Corrected C9 object SHA256 is
`0c508eb628de4b0d678df430277a01d5912086f9cc2c24674b2a9a70a3dce30c`.

Corrected C9 evaluation is Base `39.580/57.160/42.034`, Novel
`34.202/58.176/40.923`, Combined `38.944/57.280/41.903/17.649` when the
internal evaluator also exposes ClsA. Relative to the reproduced baseline,
Novel AssocA is `-0.013`; therefore the explicit V8 gain gate fails.

## Disposition

COVTrack-native PSMR training and Test PSMR were **SKIPPED_BY_NO_NOVEL_GAIN_GATE**.
This is a measured, hash-backed no-gain branch, not a missing-result or a
synthetic zero. COV official Val/Test baselines remain complete and usable as
cross-baseline controls.
