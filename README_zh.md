# EIDOS 最新审阅版论文复现包（精简版）

论文依据是 `EIDOS_ESWA_v12_reviewed.zip`，其 SHA-256 记录在
`paper/source.json`。本包可独立解压运行，不依赖原项目目录、原虚拟环境、
GPU、账号或其他符号回归方法。

本版约 26 MB，保留全部实验实际使用的数据与全部已纳入的实验。
完整 PMLB 源数据改为下载链接及重建脚本；安装依赖后，默认复现仍可离线运行。
精简后的独立运行验证见 `validation/COMPACT_VALIDATION.md`。

版本核对发现：主预测结果与现行代码一致，但论文的恢复曲线、区间统计使用了较早的
实验实现。本包保留实际生成这些结果的归档源码，并在 `docs/PROTOCOL_NOTES.md`
明确列出与后来协议修订的差异；不将两种协议混称为同一实现。

## 从零开始

安装 **64 位 Python 3.12**，解压本包，在包目录执行：

Windows：

```powershell
py -3.12 bootstrap.py --suite smoke
py -3.12 bootstrap.py --suite all --workers 4
```

Linux / macOS：

```sh
python3.12 bootstrap.py --suite smoke
python3.12 bootstrap.py --suite all --workers 4
```

脚本自动创建包内 `.venv`，安装锁定版本的依赖，再运行实验。安装依赖需要联网；
默认主实验所需数据全部在包内，安装后可离线运行。也可用绝对路径从任意工作目录
调用脚本。`smoke` 是小规模端到端检查，完整复现使用 `all`。

## 复现范围

| 内容 | 规模 | 输出 |
|---|---|---|
| PMLB 预测与隔离检查 | 243 个任务，包括 200 个机制任务和 43 个表格任务；每个任务额外替换后续角色数据重跑一次 | 逐任务 RMSE、预测区间、模型表达式、参数、解码一致性和替换不变性 |
| OpenML 预测 | 15 个数据集，每个 10 次划分，共 150 次拟合 | 逐次结果及逐数据集汇总 |
| 方程恢复 | 5 个生成机制 × 8 个噪声水平 × 5 个种子，共 200 次拟合 | 精确结构恢复率、语义 NRMSE、算子精确率/召回率、三个代表公式 |
| 区间有效性 | 5 个机制 × 2 种分布设置 × 5 个种子，共 50 次拟合 | EIDOS 预测上的 pooled、Mondrian 和辅助 proxy-trend 区间，共 125 行统计 |

不包含对照方法、消融实验、不同数据分配方案比较、外部公式编解码和大规模
注入泄漏实验。因此，本包不计算依赖其他方法的排名、胜出次数、相对 RMSE 和
显著性检验，也不声称覆盖整篇论文的所有结果。

主预测实验使用论文最终 EIDOS 配置。受控恢复实验采用统一的 240 个候选结构库；
区间实验保留论文对应的归档配置和输入。三者是论文原有的不同实验协议，不混用。

## 结果和验证

完整运行输出在 `results/`，小规模检查输出在 `results_smoke/`：

- `verification_report.json`：与归档论文结果逐项核对，检查任务完整性、模型解码和数据隔离。
- `summary.json`、各实验的 `raw.csv` 和 `summary.csv`：本次实际计算结果。
- `benchmark/artifacts/`、`openml/artifacts/`：数据角色、预处理参数、模型记录和逐点测试预测。
- `recovery/artifacts/`：合成观测、候选评分账本、参数和逐点预测。
- `recovery/formula_examples.csv`：三个 EIDOS 代表公式；恢复率与区间覆盖率图同时输出 PNG、PDF。
- `environment.json`：实际 Python、依赖、系统和代码/输入指纹。

所有模型均重新拟合。`reference/` 只在拟合之后用于比对，不参与数据生成、选择或训练。
`validation/` 保存本包交付前的验证记录，不是运行缓存。失败会保留并返回非零退出码。
数值比较使用 `rtol=1e-5, atol=1e-8`；候选身份、恢复标记采用精确比较。
PMLB 的 RMSE 和区间宽度另允许 `atol=1e-5`，相当于目标量化步长 `0.001`
的 1%；更严格的 `1e-8` 匹配数及差异仍保留。Linux 全量验证仅一个任务有
约 `4.12e-7` 的 RMSE 差异和 `2.01e-6` 的宽度差异，所有候选身份和覆盖率均一致。
运行时间按当前机器重新测量，不要求等于论文的原机器时间。

PMLB/OpenML 的 `rmse` 使用发现集标准化的目标单位，同时提供 `rmse_raw_units`。
语义 NRMSE 以测试集无噪声真值的标准差归一化。模型记录包含完整基函数标准化状态，
`repro.records.unpack` 可以不经训练直接重建预测器。跨环境数值容差与同环境字节不变性
分别验证，不把不同 BLAS 环境的浮点字节相等作为可移植性的前提。

## 实验数据与完整源数据下载

所有 243 个 PMLB 任务的精确实验样本均已打包。完整 PMLB 源数据不放入精简包，
全部 243 个下载链接列在 `data/pmlb/DOWNLOADS.md`，并提供清洗、确定性抽样和
一致性验证脚本。默认实验直接使用包内精确样本，不需要下载完整原始表。

15 个 OpenML 数据集的完整数值清洗表、区间实验输入和 200 份方程恢复数据也已包含。
所有数据采用可移植的 CSV/JSON，没有依赖 Python 版本的 pickle。
恢复实验从固定方程和种子重新生成观测，所附 200 份 CSV 用于核查，不作为训练缓存。

准备脚本可以从原始数据重新清洗、抽样并检查一致性。以下 `python` 应指向本包的
`.venv/Scripts/python.exe`（Windows）或 `.venv/bin/python`（Linux/macOS）：

```sh
python prepare_data.py --track pmlb --download-missing --output rebuilt_data
python prepare_data.py --track openml --from-source --output rebuilt_data
python prepare_data.py --track recovery --output rebuilt_data
```

完整来源、文件大小和校验值位于 `data/pmlb/manifest.json`；大文件下载说明位于
`data/LARGE_DATA.md`。准备脚本会报告精确数值匹配或 `1e-12` 容差内匹配，发现输入
不一致时不会覆盖归档数据。默认实验使用包内精确样本。

`--download-missing` 将源表下载到 `rebuilt_data/downloads/` 并在后续运行复用。
不指定数据集时会重建全部 243 个源表，需要联网及额外磁盘空间。也可先核查两个任务：

```sh
python prepare_data.py --track pmlb --datasets 706_sleuth_case1202,feynman_I_6_2a --download-missing --output rebuilt_data
```

## 续跑与单独运行

```sh
python reproduce.py --suite all --workers 4 --resume
python reproduce.py --suite recovery --workers 4 --output results_recovery
python -m unittest discover -s tests -v
python verify_manifest.py
```

`--resume` 仅复用代码、配置及输入指纹一致的成功单元。默认不启用续跑，确保实际重新拟合。
完整的实验对应关系见 `docs/PROVENANCE.md`，本次实测范围和结果见
`validation/VALIDATION.md`。英文说明另含 Docker 用法和输出格式细节。
