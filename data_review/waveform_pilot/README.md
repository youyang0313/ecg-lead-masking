# PTB-XL 训练数据来源与小样本波形核查

日期：2026-10-23。这是输入可行性核查，不是模型性能实验。

## 官方来源

数据为 [PhysioNet PTB-XL 1.0.3](https://physionet.org/content/ptb-xl/1.0.3/)，版本 DOI 为 [10.13026/kfzx-aw45](https://doi.org/10.13026/kfzx-aw45)。原始数据集论文为 [Wagner et al., 2020](https://doi.org/10.1038/s41597-020-0495-6)。本项目固定采用该版本的 `records100`：10 秒、100 Hz、12 导联，每条记录 12×1000 个采样值。

实际下载使用官方页面列出的公开 S3 存储桶 `s3://physionet-open/ptb-xl/1.0.3/`，HTTP 前缀为 `https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/`。没有采用第三方重新打包数据。原始文件保存在本目录 `originals/`，每个文件的 URL、字节数和 SHA-256 保存在 [sources.json](D:/AAAurora/NTUDL/ECGProject/data_review/waveform_pilot/sources.json)。完整元数据与许可证位于上一级 `metadata/`。

正式研究计划按官方患者划分使用 folds 1–8 训练、9 验证、10 测试。目标为 NORM、MI、STTC、CD、HYP 五个非互斥诊断超类，按代码存在性聚合。这些是原有心电注释，不是独立确诊或急性心梗分诊终点。

## 抽样与检查结果

抽样在检查波形前完成，随机种子为 20261023。只从训练 folds 1–8 选择 100 个不同患者的 100 条记录，先确保每个目标至少 20 条阳性，再补足至 100 条。标签有重叠，阳性数为 NORM 39、MI 26、STTC 26、CD 21、HYP 20。这是为核查标签覆盖而构造的样本，不代表全库患病率或信号质量。固定清单见 [selection.json](D:/AAAurora/NTUDL/ECGProject/data_review/waveform_pilot/selection.json)；失败记录不会因失败而更换。

结果保存在 [audit.json](D:/AAAurora/NTUDL/ECGProject/data_review/waveform_pilot/audit.json)：

- 100/100 条记录通过读取、维度、采样率、单位、有限值及元数据对应检查；200 个原始文件哈希均核验。
- 1,200 个通道初值与 1,200 个通道校验和均与文件头一致。
- 实际文件头使用 `AVR/AVL/AVF`。读取时按名称对应到协议中的 `aVR/aVL/aVF`，并显式重排导联。
- 对每条记录检查 S12、S6、S4、S3、S2 五种配置，共 500 次：保留导联逐值不变，不可用导联在标准化后置零，形状和有限值符合要求。
- 未读取验证或测试波形，未解析测试诊断代码，未构建或训练模型。

最初核查时环境没有 WFDB 库。该核查程序依据官方 [信号格式说明](https://physionet.org/physiotools/wag/signal-5.htm)和[文件头说明](https://physionet.org/physiotools/wag/header-5.htm)，仅支持本样本使用的单文件、多通道交织、16 位小端整数格式；不支持的格式会报错。随后已在项目专用目录安装 WFDB 4.1.2，并完成 [标准读取器交叉核对](D:/AAAurora/NTUDL/ECGProject/data_review/waveform_pilot/wfdb_crosscheck.json)：100 条记录共 120 万个采样值，在统一导联顺序、单位和 float32 表示后逐值完全一致。正式数据处理采用标准 WFDB 读取器。本次校验不证明医学注释正确，也不证明 100 Hz 足以支持所有诊断。

## 输入示例

示例使用预先选定清单中的第一条 ECG 12065（训练 fold 4、NORM 注释），没有按波形外观挑选。图中不据此宣称患者健康。图源记录见 [figure_sources.json](D:/AAAurora/NTUDL/ECGProject/data_review/waveform_pilot/figures/figure_sources.json)，PNG 和 SVG 均已保存。

![训练记录的真实 12 导联波形，单位 mV](D:/AAAurora/NTUDL/ECGProject/data_review/waveform_pilot/figures/training_ecg_example.png)

下图展示五种输入配置，并以 V2 展示保留与屏蔽结果。标准化参数仅来自这 100 条训练样本，用于此次检查和作图；正式训练须使用全部合格训练记录重新估计。置零表示输入编码约定，不表示生理上的平直心电。

![导联配置及 V2 在不同配置下的输入示例](D:/AAAurora/NTUDL/ECGProject/data_review/waveform_pilot/figures/input_mask_example.png)

## 复现与后续边界

`prepare_waveform_pilot.py` 固定选择并下载原始文件，`check_waveform_pilot.py` 用 NumPy 完成核查，`plot_waveform_pilot.py` 用 Matplotlib 输出图。前两者使用现有 bundled Python，作图使用现有 Anaconda Python，均未安装新依赖。

标准读取器交叉核对已经完成。全训练集及验证集的后续处理见 [正式预处理说明](D:/AAAurora/NTUDL/ECGProject/data/ptbxl_1_0_3/README.md)。本页记录的 100 条只是可行性样本，通过检查不能替代模型训练、验证或测试结果。
