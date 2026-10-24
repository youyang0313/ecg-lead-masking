# PTB-XL 1.0.3：正式训练与验证数据准备

本目录用于输入预处理，不包含模型实现。实际完成状态以 `prepared/READY.json`、[核查结果](D:/AAAurora/NTUDL/ECGProject/data_review/preprocessing_audit.json)和[保存结果复核](D:/AAAurora/NTUDL/ECGProject/data_review/preprocessing_verification.json)为准；运行中产生的部分文件不表示预处理已完成。

2026-10-23 已完成并通过复核：训练 **17,084 条 / 14,823 位患者**，验证 **2,146 条 / 1,917 位患者**。原始文件哈希全部通过，没有新增波形排除。详见[中文发现与边界](D:/AAAurora/NTUDL/ECGProject/data_review/PREPROCESSING_REVIEW_ZH.md)。

## 来源与纳入规则

来源为 [PhysioNet PTB-XL 1.0.3](https://physionet.org/content/ptb-xl/1.0.3/)，版本 DOI 为 [10.13026/kfzx-aw45](https://doi.org/10.13026/kfzx-aw45)。官方页面列出的 S3 存储桶通过 `https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/` 访问。每条记录使用官方 `records100` 的 10 秒、100 Hz、12 导联波形，不自行重复降采样。

`development_plan.json` 固定元数据来源哈希、目标顺序、诊断标签筛选、训练/验证清单与排除记录。folds 1–8 用于训练，fold 9 用于验证。fold 10 仅用于患者划分检查，不解析其诊断代码、不下载其波形。记录有至少一个目标诊断超类才进入下载清单；334 条训练记录及 37 条验证记录因缺少目标映射排除。

`originals/` 保留原始 `.hea` 和 `.dat` 文件。`download_manifest.jsonl` 逐文件记录官方来源、获取方式、SHA-256、大小与时间，支持中断后按哈希核验续传。首次下载已保存的 100 条 pilot 数据在验证原哈希后复用。网络失败会阻止后续处理，不作为医学/信号资格排除。

## 处理方法

采用标准 WFDB 4.1.2 读取物理量。读取器已经与早期 pilot 读取器在 100 条记录的 120 万个 float32 采样值上逐值一致，见 [交叉核查](D:/AAAurora/NTUDL/ECGProject/data_review/waveform_pilot/wfdb_crosscheck.json)。

1. 核验每个原始文件哈希，再检查读取、12 条唯一预期导联、100 Hz、1,000 个采样点、mV 单位及有限值。按实际导联名称重排；`AVR/AVL/AVF` 对应 `aVR/aVL/aVF`。不合格记录保存至 `prepared/waveform_exclusions.json`，所有后续模型和输入配置使用共同资格集合。
2. 标签按诊断代码存在性映射为 NORM、MI、STTC、CD、HYP 五维向量，不将 likelihood 当作软概率，不把 NORM 改成其他类别的补集。
3. 仅在全部合格、完整的训练波形上，以 float64 累积每条导联的均值和总体标准差。标准差下限为 `1e-6 mV`。参数及拟合记录清单保存为 `prepared/scaler.json` 和 `scaler_training_ecg_ids.json`。
4. 训练与验证信号复用训练参数标准化，保存为 float32。每条记录的五种预设屏蔽配置均检查保留导联不变、缺失通道置零；不在屏蔽后重新标准化，也不按保留比例放大。
5. 主实验不额外滤波、去噪或切分心搏。元数据中的噪声标记和常量导联本身不单独触发排除，常量导联计数另行报告。当前处理并不证明每条记录的医学注释正确，也不证明其临床适用性。

## 输出与使用

`prepared/train.signals.npy` 和 `validation.signals.npy` 的形状为 `[记录数, 12, 1000]`，已完成标准化但保留全部 12 条导联；训练时随机屏蔽和评价时固定屏蔽应在读取之后施加。它们可用 NumPy 的 `mmap_mode="r"` 加载，避免一次占用全部内存。

同名 `*.labels.npy` 为 uint8、形状 `[记录数, 5]`，列顺序固定为 `NORM, MI, STTC, CD, HYP`。`*.records.json` 的行与信号、标签严格对应，保留 ECG ID、患者 ID、fold、原始路径与标签。患者 ID 不作为模型输入。

`READY.json` 仅在输出、统计与检查全部完成后写入，记录产物哈希。独立 `verify` 步骤核对所有输出哈希、患者划分及 scaler 拟合清单，并从训练/验证各均匀选取 20 条记录，重新读取原始数据以核查持久化结果与标签对应。

## 复现命令

本次使用已有 `D:/py/Anaconda3/python.exe`（Python 3.9.7）。WFDB 4.1.2 安装于项目内 `ECGProject/.runtime/data-site-packages`，没有升级现有全局依赖。版本清单见 [requirements-data.txt](D:/AAAurora/NTUDL/ECGProject/data_review/requirements-data.txt)。从 `D:/AAAurora/NTUDL` 的 PowerShell 顺序执行：

```powershell
& 'D:/py/Anaconda3/python.exe' 'ECGProject/data_review/test_preprocessing.py'
& 'D:/py/Anaconda3/python.exe' 'ECGProject/data_review/preprocess_development.py' pilot
& 'D:/py/Anaconda3/python.exe' 'ECGProject/data_review/prepare_development.py' plan
& 'D:/py/Anaconda3/python.exe' 'ECGProject/data_review/prepare_development.py' fetch --workers 24
& 'D:/py/Anaconda3/python.exe' 'ECGProject/data_review/preprocess_development.py' process
& 'D:/py/Anaconda3/python.exe' 'ECGProject/data_review/preprocess_development.py' verify
```

已完成的数据可直接执行 `verify`；`process` 会拒绝覆盖已有 `READY.json` 的结果。若重新定义标签、资格或标准化方案，应先记录版本变更，再建立新的输出版本，不静默覆盖当前分析依据。
