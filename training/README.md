# ECG 模型实现与工程验证

2026-10-23：**正式六模型训练和独立 fold-10 测试均已完成并通过复核。**[独立测试结果](D:/AAAurora/NTUDL/ECGProject/training/INDEPENDENT_TEST_REVIEW_ZH.md)记录 2,158 条测试记录、1,877 位患者与 2,000 次患者成对 bootstrap。主效果为 +0.19745 AUROC，95% 区间 [0.18816, 0.20670]，条件于已训练种子集合。[训练完成记录](D:/AAAurora/NTUDL/ECGProject/training/MAIN_TRAINING_REVIEW_ZH.md)包含验证集汇总、实际成本及 Windows 日志保存错误的恢复过程。训练运行目录为 `experiments/train/20261022T183345197538Z/`，测试运行目录为 `experiments/test/20261022T202402963049Z/`。

## 模型与控制条件

`model.py` 按 [protocol_v1.json](D:/AAAurora/NTUDL/ECGProject/report/protocol_v1.json) 实现紧凑型 1D 残差 CNN，359,749 个可训练参数。输入 `[N,12,1000]`，输出五个 logits，使用未加权 BCEWithLogitsLoss。A 始终完整输入；B 以 50% 概率保留完整输入，否则均匀选择保留数量 1–11，再均匀选择该数量的导联子集。mask 沿时间轴固定，标签不变。

`runtime.py` 检查已准备数据的哈希并以只读内存映射读取；批次复制后才施加 mask。A/B 重置相同初始化种子，样本顺序由独立 NumPy 随机流按 epoch 决定，mask 使用独立 CPU torch.Generator。验证为 float32、冻结模型参数和 BN 运行统计，五种配置分别计算指标，以协议中的统一权重选一个检查点；相同分数保留更早 epoch。

训练使用现有 GPU 环境 `D:/py/Anaconda3/envs/fl_pytorch/python.exe`：Python 3.9.22、PyTorch 2.6.0、NumPy 2.0.2、CUDA build 12.6、cuDNN 9.8、RTX 4060 Laptop GPU。没有新增或升级该环境的依赖。`run_gpu.ps1` 只为调用进程配置 DLL 搜索路径，结束时恢复 PATH。

共同数值策略为 float16 CUDA autocast + 动态 GradScaler，参数保持 float32；禁用 TF32 和 cuDNN benchmark，启用确定性算法。梯度溢出会由 GradScaler 跳过并记录，非有限损失直接报错；没有梯度裁剪或额外输入增强。本次短程运行没有溢出或跳过的优化步骤。

## 已保存的检查

- [基础检查](D:/AAAurora/NTUDL/ECGProject/training/engineering_checks.json)：相同初始化、完整/部分输入分布、非空 mask、随机流隔离、整导联屏蔽、真实数据前向/反向、稳定且不修改状态的评价、选模权重/并列规则与学习率端点。
- [指标交叉核对](D:/AAAurora/NTUDL/ECGProject/training/metric_reference_check.json)：300 组二分类、20 组多标签，包含并列分数；AUROC 和非插值 AP 与现有 scikit-learn 0.24.2 的最大绝对误差为 2.22e-16。GPU 环境无需安装 scikit-learn。
- [最新短程运行入口](D:/AAAurora/NTUDL/ECGProject/training/latest_smoke.json)：清单包含选定 ECG ID、版本、源代码哈希及环境；各组目录保存训练历史、最佳检查点、验证概率和摘要。源代码与原始设计文件另存于 `source/`。恢复代码时应还原原有项目目录结构，数据仍由上述完整性检查约束。
- [归档复核](D:/AAAurora/NTUDL/ECGProject/training/engineering_verification.json)：64 次有限梯度更新、实际可训练权重变化、A/B 初始化与样本顺序一致、检查点回读预测逐值一致、两次短程运行的选定检查点预测一致。

试运行使用 1,024 条训练记录（固定随机样本加两条数值压力记录），每组两轮，验证用完整 fold 9 的 2,146 条记录。它只检查工程行为，不能据其分数检验研究假设。数据选择、精度、数值压力与资源结果见[中文记录](D:/AAAurora/NTUDL/ECGProject/training/ENGINEERING_REVIEW_ZH.md)。

## 执行

从 `D:/AAAurora/NTUDL` 的 PowerShell 运行：

```powershell
# 已完成：只运行基础工程检查
& 'ECGProject/training/run_gpu.ps1' check
# 已完成：A/B 各两轮的短程工程试运行，独立目录保存
& 'ECGProject/training/run_gpu.ps1' smoke
# 只复核已保存结果，不训练
& 'D:/py/Anaconda3/python.exe' 'ECGProject/training/verify_engineering.py'
```

正式实验入口为 `& 'ECGProject/training/run_gpu.ps1' train`，本轮已执行 seeds 17/29/43 × A/B × 40 epochs，使用全部合格训练记录。一次日志保存错误导致 seed-43 A 从头重跑，原尝试仍保留，另计 40 轮成本；恢复驱动为 `recover_main.py`。正式模式不加载试运行检查点，已有目录不被覆盖。

`verify_main.py` 使用同一 PyTorch 2.6 环境执行只读复核并生成分析文件，已通过。[选定模型清单](D:/AAAurora/NTUDL/ECGProject/experiments/train/20261022T183345197538Z/analysis/selected_models.json)记录六个模型、参数哈希与固定配置；[验证集汇总](D:/AAAurora/NTUDL/ECGProject/experiments/train/20261022T183345197538Z/analysis/validation_summary.json)属于选模数据上的描述性结果。

## 已完成的独立测试

`data_review/prepare_test.py plan|fetch|process|verify` 在固定检查点后显式释放测试数据；其原始文件、计划、标准化数组与审计独立保存，没有改写开发数据。`evaluate_test.py` 在上述 GPU 环境中运行固定模型，复用原有 float32 评价函数并校验模型状态。`analyze_test.py` 在基础 Python 环境计算统计，`test_statistics.py` 实现等价于患者记录重复展开的加权 AUROC。`verify_test.py` 使用既有 scikit-learn 独立核对保存指标、三个展开抽样及区间，`plot_test_results.py` 从复核通过的结果生成报告图。全部入口本轮已执行，不需重新评价测试集来修改模型。

[最新测试入口](D:/AAAurora/NTUDL/ECGProject/training/latest_test.json)指向预测、执行清单、测试前源码快照及 `analysis/` 内统计与复核结果。运行汇总包括完整输入变化；次要区间为描述性分析，不能视为临床等效或安全验证。

原始协议 JSON 中“未实现/暂停”的状态属于写作时记录，为保留数据来源哈希未修改；当前执行状态与精度细节见 [execution_plan.json](D:/AAAurora/NTUDL/ECGProject/training/execution_plan.json)。
