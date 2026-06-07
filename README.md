# 4D Transmission Risk Predictor

SVD(128) + ExtraTrees regression model for predicting four-dimensional transmission risk of *Acinetobacter baumannii*.

- Input: Whole-genome SNP variants (151,913 sites) + Environmental factors
- Output: Network Hub / Clone Advantage / Persistence / Spatial Connectivity

Target pathogen: *Acinetobacter baumannii* (Aba), a WHO priority multidrug-resistant pathogen.

[English](#overview) | [中文说明](#中文说明)

---

## Overview

This toolkit predicts four-dimensional transmission risk scores for *A. baumannii* isolates:

| Dimension | Description | Primary Driver |
|:----------|:------------|:---------------|
| **Network Hub** | Network centrality in transmission network | Genomic (99.1%) |
| **Clone Advantage** | Competitive clone advantage | Genomic (97.0%) |
| **Persistence** | Sustained transmission capability | Genomic (92.7%) |
| **Spatial Connectivity** | Cross-regional spread potential | Environmental (81.0%) |

- Training samples: 689 isolates
- SNP variants: 151,913 (full genome)
- SVD components: 128 (99.48% explained variance)
- Environmental features: PM2.5, PM10, SO2, NO2, CO, O3, AQI

## Installation

```bash
git clone https://github.com/kevin2000s/4D-Risk-Predictor.git
cd 4D-Risk-Predictor
pip install -r requirements.txt
```

Requirements: Python >= 3.9, 4GB+ RAM (8GB recommended)

## Build Executable

To build a standalone `.exe` (Windows) or binary (Linux/macOS):

**Windows:**
```cmd
build.bat
```

**Linux / macOS:**
```bash
chmod +x build.sh
./build.sh
```

The executable will be at `dist/4D_Risk_Predictor/4D_Risk_Predictor.exe`.

## Quick Start

### 1. Download Pre-trained Models

Model files (`*.joblib`) are not stored in this repository. Download `models.zip` from the [Releases](https://github.com/kevin2000s/4D-Risk-Predictor/releases) page and extract to the project root.

```bash
wget https://github.com/kevin2000s/4D-Risk-Predictor/releases/download/v1.0/models.zip
unzip models.zip
```

Contents of `models.zip`:

| File | Size | Description |
|:-----|:-----|:------------|
| `svd128_extratrees_models.joblib` | ~19 MB | ExtraTrees regressors (4 dimensions) |
| `svd128_svd_transformer.joblib` | ~74 MB | TruncatedSVD transformer (128 components) |
| `svd128_model_metadata.joblib` | ~2 MB | SNP IDs and feature names |
| `svd128_env_scaler.joblib` | ~1 MB | Environment scaler |

### 2. Run Prediction

**GUI:**
```bash
python main.py
```

**CLI:**
```bash
# Full prediction (SNP + environment)
python prediction_toolkit/predict.py \
    --snp data/snp.csv \
    --env data/env.csv \
    --out predictions.csv

# Environment-only prediction
python prediction_toolkit/predict.py \
    --env data/env.csv \
    --out predictions.csv \
    --env-only

# Batch prediction
python prediction_toolkit/batch_predict.py \
    --snp-dir data/snp_batches/ \
    --env-file data/env.csv \
    --out results.csv
```

**Python API:**
```python
from prediction_toolkit.model import TransmissionRiskPredictor
from prediction_toolkit.data_loader import SNPDataLoader, EnvDataLoader, align_samples

predictor = TransmissionRiskPredictor()

snp_loader = SNPDataLoader(predictor.snp_ids)
X_snp, snp_samples = snp_loader.load_long_format('snp_data.csv')

env_loader = EnvDataLoader()
X_env, env_samples = env_loader.load('env_data.csv')

X_snp_a, X_env_a, samples = align_samples(X_snp, snp_samples, X_env, env_samples)
results = predictor.predict(X_snp_a, X_env_a, samples)
```

## Data Format

### SNP Data (Long Format)

| Column | Description | Example |
|:-------|:------------|:--------|
| `CHROM` | Chromosome / contig ID | `1` |
| `POS` | Position (1-based) | `527498` |
| `TYPE` | Variant type | `snp` |
| `REF` | Reference allele | `A` |
| `ALT` | Alternate allele | `C` |
| `sample_count` | Number of samples with this SNP | `2` |
| `sample_ids` | Space-separated sample IDs | `EA10489 EA10561` |

Example (`prediction_toolkit/example/example_snp.csv`):

```csv
CHROM,POS,TYPE,REF,ALT,sample_count,sample_ids
1,527498,snp,A,C,2,EA10489 EA10561
1,1520479,snp,G,A,1,EA10641
```

### Environment Data

| Column | Description | Unit |
|:-------|:------------|:-----|
| `sample_id` | Sample identifier | - |
| `PM2.5` | Fine particulate matter | ug/m3 |
| `PM10` | Coarse particulate matter | ug/m3 |
| `SO2` | Sulfur dioxide | ug/m3 |
| `NO2` | Nitrogen dioxide | ug/m3 |
| `CO` | Carbon monoxide | mg/m3 |
| `O3` | Ozone | ug/m3 |
| `AQI` | Air Quality Index | - |

Example (`prediction_toolkit/example/example_env.csv`):

```csv
sample_id,PM2.5,PM10,SO2,NO2,CO,O3,AQI
EA10489,9.0,10.0,2.0,13.0,0.6,62.0,20.0
```

Missing values are automatically filled with column means.

## Model Performance

| Dimension | 5-Fold CV R2 | Test R2 | Test MAE | Primary Driver |
|:----------|:------------:|:-------:|:--------:|:---------------|
| Network Hub | 0.884 +- 0.017 | 0.901 | 0.055 | Genomic |
| Clone Advantage | 0.797 +- 0.047 | 0.868 | 0.047 | Genomic |
| Persistence | 0.821 +- 0.055 | 0.900 | 0.044 | Genomic |
| Spatial Connectivity | 0.942 +- 0.021 | 0.927 | 0.034 | Environmental |

- Overall feature contribution: Genomic 77.0% | Environmental 23.0%
- SVD explained variance: 99.48% (128 components)

## File Structure

```
4D-Risk-Predictor/
├── main.py                           # GUI entry
├── build.bat                         # Windows build
├── build.sh                          # Linux/macOS build
├── prediction_toolkit/               # Core package
│   ├── __init__.py
│   ├── predict.py                    # CLI
│   ├── batch_predict.py              # Batch prediction
│   ├── model.py                      # Model wrapper
│   ├── data_loader.py                # Data loaders
│   ├── vcf_parser.py                 # VCF parser
│   └── example/                      # Example data
│       ├── example_snp.csv
│       └── example_env.csv
├── svd128_extratrees_feature_importance.csv
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── README.md

# Pre-trained model files (~96 MB) -- download from Releases
#   svd128_extratrees_models.joblib   # ExtraTrees regressors
#   svd128_svd_transformer.joblib     # TruncatedSVD transformer
#   svd128_env_scaler.joblib          # Environment scaler
#   svd128_model_metadata.joblib      # SNP IDs and feature names
```

## Citation

If you use this software in your research, please cite:

> SVD(128) + ExtraTrees 4D Transmission Risk Prediction Model for *Acinetobacter baumannii*.
> Training: 689 A. baumannii isolates, 151,913 SNPs, scikit-learn 1.3.0.

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## 中文说明

本软件基于 SVD(128) + ExtraTrees 回归模型，预测鲍曼不动杆菌（*Acinetobacter baumannii*, Aba）的四维传播风险评分。

鲍曼不动杆菌（Aba）是医院获得性感染的重要致病菌。本模型通过整合全基因组 SNP 变异和环境因素，对 Aba 分离株的传播风险进行四维量化评估。

### 快速开始

**1. 下载预训练模型**

模型文件（`*.joblib`）不存储在代码仓库中。请从 [Releases](https://github.com/kevin2000s/4D-Risk-Predictor/releases) 页面下载 `models.zip` 并解压到项目根目录。

```bash
wget https://github.com/kevin2000s/4D-Risk-Predictor/releases/download/v1.0/models.zip
unzip models.zip
```

`models.zip` 内含文件：

| 文件 | 大小 | 说明 |
|:-----|:-----|:-----|
| `svd128_extratrees_models.joblib` | ~19 MB | ExtraTrees 回归模型（4 个维度） |
| `svd128_svd_transformer.joblib` | ~74 MB | SVD 降维转换器（128 个主成分） |
| `svd128_model_metadata.joblib` | ~2 MB | SNP ID 及特征名 |
| `svd128_env_scaler.joblib` | ~1 MB | 环境数据标准化器 |

**2. 运行预测**

```bash
# 安装依赖
pip install -r requirements.txt

# GUI 模式
python main.py

# 命令行模式
python prediction_toolkit/predict.py --snp data/snp.csv --env data/env.csv --out result.csv
```

### 模型性能

| 维度 | 5折交叉验证 R2 | 测试集 R2 | 主要驱动因素 |
|:----------|:------------:|:-------:|:---------------|
| Network Hub（网络中心性）| 0.884 +- 0.017 | 0.901 | 基因组 (99.1%) |
| Clone Advantage（克隆优势）| 0.797 +- 0.047 | 0.868 | 基因组 (97.0%) |
| Persistence（持续传播能力）| 0.821 +- 0.055 | 0.900 | 基因组 (92.7%) |
| Spatial Connectivity（空间连通性）| 0.942 +- 0.021 | 0.927 | 环境 (81.0%) |

详细数据格式说明和 API 用法见上方英文部分。
