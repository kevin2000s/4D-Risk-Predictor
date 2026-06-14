#!/usr/bin/env python3
"""
VCF 转 CSV 转换器

将 Snippy 或其他工具生成的 VCF 文件转换为 4D Risk Predictor 需要的长格式 CSV：
    CHROM,POS,TYPE,REF,ALT,sample_count,sample_ids

支持三种输入方式：
    1. 单个多样本 VCF (--input combined.raw.vcf)
    2. 多个单样本 VCF 文件 (--input "sample_*/snps.vcf")
    3. Snippy 输出目录 (--snippy-dir snippy_outputs/)

示例：
    python vcf_to_csv.py --input combined.raw.vcf --out snp_data.csv
    python vcf_to_csv.py --input "*/snps.vcf" --out snp_data.csv
    python vcf_to_csv.py --snippy-dir snippy_outputs/ --out snp_data.csv
"""
import argparse
import glob
import gzip
import os
import sys

import joblib
import pandas as pd

# 添加工具包路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


METADATA_FILE = "svd128_model_metadata.joblib"


def _load_reference_snp_ids(model_dir=None):
    """加载训练时使用的 SNP ID 列表（格式：CHROM_POS）。"""
    if model_dir is None:
        model_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    metadata_path = os.path.join(model_dir, METADATA_FILE)
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"找不到模型元数据文件: {metadata_path}\n"
            f"请确保 {METADATA_FILE} 存在于项目根目录，或用 --model-dir 指定模型文件所在目录。"
        )

    metadata = joblib.load(metadata_path)
    snp_ids = metadata.get("snp_ids", [])
    print(f"[INFO] 加载参考 SNP 列表: {len(snp_ids):,} 个位点")
    return set(snp_ids)


def _open_vcf(vcf_path):
    """根据后缀自动选择 gzip 或普通文本打开方式。"""
    if vcf_path.endswith(".gz"):
        return gzip.open(vcf_path, "rt")
    return open(vcf_path, "r")


def _parse_gt(gt_val):
    """解析基因型字段，返回是否含有 ALT 等位基因。"""
    if gt_val in (".", "./.", ".|."):
        return False
    for allele in gt_val.replace("|", "/").split("/"):
        if allele not in ("0", ".") and allele != "":
            return True
    return False


def _parse_vcf_samples(vcf_path, ref_snp_ids):
    """
    解析单个 VCF 文件，返回每个 SNP 上携带 ALT 的样本集合。

    Returns
    -------
    dict : {snp_id: {"CHROM": str, "POS": str, "REF": str, "ALT": str, "samples": set}}
    """
    snp_data = {}

    with _open_vcf(vcf_path) as f:
        sample_names = []
        gt_start_col = -1

        for line in f:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                parts = line.strip().split("\t")
                sample_names = parts[9:]
                gt_start_col = 9
                continue
            if line.startswith("#"):
                continue

            parts = line.strip().split("\t")
            if len(parts) < gt_start_col + 1:
                continue

            chrom = parts[0]
            pos = parts[1]
            ref = parts[3]
            alt = parts[4]
            fmt = parts[8]

            # 只保留双等位 SNP
            if len(ref) != 1 or len(alt) != 1:
                continue

            snp_id = f"{chrom}_{pos}"
            if snp_id not in ref_snp_ids:
                continue

            # 找到 GT 在 FORMAT 中的位置
            fmt_fields = fmt.split(":")
            gt_idx = fmt_fields.index("GT") if "GT" in fmt_fields else 0

            alt_samples = set()
            for sample_name, sample_gt in zip(sample_names, parts[gt_start_col:]):
                gt_val = sample_gt.split(":")[gt_idx] if ":" in sample_gt else sample_gt
                if _parse_gt(gt_val):
                    alt_samples.add(sample_name)

            if not alt_samples:
                continue

            if snp_id in snp_data:
                snp_data[snp_id]["samples"].update(alt_samples)
            else:
                snp_data[snp_id] = {
                    "CHROM": chrom,
                    "POS": int(pos),
                    "REF": ref,
                    "ALT": alt,
                    "samples": alt_samples,
                }

    return snp_data


def _collect_snippy_vcf_files(snippy_dir):
    """
    在 Snippy 输出目录中收集每个样本的 snps.vcf 文件。

    期望结构：
        snippy_dir/
        ├── sample_A/
        │   └── snps.vcf
        ├── sample_B/
        │   └── snps.vcf
    """
    vcf_files = []
    for root, dirs, files in os.walk(snippy_dir):
        if "snps.vcf" in files:
            sample_name = os.path.basename(root)
            vcf_files.append((sample_name, os.path.join(root, "snps.vcf")))

    if not vcf_files:
        # 尝试其他命名模式
        for vcf_path in glob.glob(os.path.join(snippy_dir, "*", "*.vcf")):
            sample_name = os.path.basename(os.path.dirname(vcf_path))
            vcf_files.append((sample_name, vcf_path))

    return vcf_files


def _build_long_format(snp_data):
    """将 SNP 数据字典转换为长格式 DataFrame。"""
    rows = []
    for snp_id in sorted(snp_data.keys(), key=lambda x: (snp_data[x]["CHROM"], snp_data[x]["POS"])):
        info = snp_data[snp_id]
        samples = sorted(info["samples"])
        rows.append({
            "CHROM": info["CHROM"],
            "POS": info["POS"],
            "TYPE": "snp",
            "REF": info["REF"],
            "ALT": info["ALT"],
            "sample_count": len(samples),
            "sample_ids": " ".join(samples),
        })

    df = pd.DataFrame(rows, columns=[
        "CHROM", "POS", "TYPE", "REF", "ALT", "sample_count", "sample_ids"
    ])
    return df


def convert_multisample_vcf(vcf_path, ref_snp_ids):
    """转换单个多样本 VCF 文件。"""
    print(f"[INFO] 解析多样本 VCF: {vcf_path}")
    snp_data = _parse_vcf_samples(vcf_path, ref_snp_ids)
    return _build_long_format(snp_data)


def convert_vcf_list(vcf_files, ref_snp_ids):
    """转换多个单样本 VCF 文件。"""
    print(f"[INFO] 解析 {len(vcf_files)} 个 VCF 文件")

    aggregated = {}
    for vcf_path in vcf_files:
        print(f"       {os.path.basename(vcf_path)}")
        per_vcf = _parse_vcf_samples(vcf_path, ref_snp_ids)
        for snp_id, info in per_vcf.items():
            if snp_id in aggregated:
                aggregated[snp_id]["samples"].update(info["samples"])
            else:
                aggregated[snp_id] = info

    return _build_long_format(aggregated)


def convert_snippy_folder(snippy_dir, ref_snp_ids):
    """转换 Snippy 输出目录。"""
    print(f"[INFO] 扫描 Snippy 输出目录: {snippy_dir}")
    vcf_files = _collect_snippy_vcf_files(snippy_dir)

    if not vcf_files:
        raise ValueError(f"在 {snippy_dir} 中未找到 snps.vcf 文件")

    print(f"       找到 {len(vcf_files)} 个样本")

    aggregated = {}
    for sample_name, vcf_path in vcf_files:
        per_vcf = _parse_vcf_samples(vcf_path, ref_snp_ids)
        for snp_id, info in per_vcf.items():
            # 用文件夹名作为样本名（覆盖 VCF 里可能的样本名）
            info["samples"] = {sample_name}
            if snp_id in aggregated:
                aggregated[snp_id]["samples"].add(sample_name)
            else:
                aggregated[snp_id] = info

    return _build_long_format(aggregated)


def parse_args():
    parser = argparse.ArgumentParser(
        description="将 VCF 文件转换为 4D Risk Predictor 所需的长格式 CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单个多样本 VCF
  python vcf_to_csv.py --input combined.raw.vcf --out snp_data.csv

  # 多个单样本 VCF（通配符）
  python vcf_to_csv.py --input "sample_*/snps.vcf" --out snp_data.csv

  # Snippy 输出目录
  python vcf_to_csv.py --snippy-dir snippy_outputs/ --out snp_data.csv

  # 指定模型文件目录
  python vcf_to_csv.py --snippy-dir snippy_outputs/ --model-dir /path/to/models --out snp_data.csv
        """
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", type=str,
                             help="输入 VCF 文件路径，支持通配符（如 '*.vcf'）")
    input_group.add_argument("--snippy-dir", type=str,
                             help="Snippy 输出目录（每个子目录包含 snps.vcf）")

    parser.add_argument("--out", type=str, required=True,
                        help="输出 CSV 文件路径")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="模型文件所在目录（默认：项目根目录）")

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("VCF 转 CSV 转换器")
    print("=" * 60)

    ref_snp_ids = _load_reference_snp_ids(args.model_dir)

    if args.input:
        vcf_files = sorted(glob.glob(args.input))
        if not vcf_files:
            raise FileNotFoundError(f"找不到匹配的 VCF 文件: {args.input}")

        if len(vcf_files) == 1:
            df = convert_multisample_vcf(vcf_files[0], ref_snp_ids)
        else:
            df = convert_vcf_list(vcf_files, ref_snp_ids)
    else:
        df = convert_snippy_folder(args.snippy_dir, ref_snp_ids)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 60)
    print(f"转换完成: {args.out}")
    print("=" * 60)
    print(f"  总 SNP 位点数: {len(df):,}")
    print(f"  总样本数: {len(set(' '.join(df['sample_ids']).split())):,}")
    print(f"\n下一步可运行:")
    print(f"  python prediction_toolkit/predict.py --snp {args.out} --env env.csv --out predictions.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
