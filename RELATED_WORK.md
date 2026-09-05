# TempoTrack 相关工作与实现边界

本项目把附件中列出的论文和代码作为设计约束与对照来源，而不是把相似关键词写成新颖性。I-JEPA、T-JEPA、GOT-JEPA 和 LeJEPA/SIGReg 说明了 latent prediction、trajectory prediction、teacher-target 与 anti-collapse 的近邻；TempoTrack 的 S1 只在冻结检测特征上预测跨断裂 tracklet identity representation，并且必须经过固定候选图和合法 path cover。它不宣称首次 JEPA tracking，也没有复制这些仓库的视觉主干或代码。

Flow Matching、DiffMOT、DiffusionTrack、DIFUSCO 和 PermFlow 分别提供向量场、运动扩散、图扩散或排列流的参照。S2 是单个源片段的 successor state distribution；S3 是可变节点数、birth/death 约束下的稀疏联合连接图；S4 是独立的 VP DDPM epsilon objective。三者不共享一个输出头，不把 MOT 图伪装成方阵排列，也不把图扩散的 TSP/MIS 解码器直接搬来。

Neural MOT solver、MOTIP、CPC 和通用记忆/RL 跟踪工作说明图关联、身份预测和记忆控制都已有先例。因此本文的可检验主张只能是：在同一固定观测、候选、预算和合法求解器下，预测监督或图生成是否改善断裂身份重连；若没有真实 trial/full artifact，就不宣称改善。

详细 URL、commit、许可证和不能复用的部分见 [`docs/research/source_ledger.json`](docs/research/source_ledger.json)。
