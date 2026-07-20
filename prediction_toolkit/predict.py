#!/usr/bin/env python3
# Copyright (c) 2025 Peking University People's Hospital Hui Lab
# SPDX-License-Identifier: MIT
"""Command-line interface for 4D transmission risk prediction."""
import argparse
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import TransmissionRiskPredictor
from data_loader import SNPDataLoader, EnvDataLoader, align_samples


def parse_args():
    parser = argparse.ArgumentParser(
        description='SVD(128) + ExtraTrees 4D 传播风险预测',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
数据格式说明:

  SNP数据（长格式）:
    CSV文件，列: CHROM, POS, TYPE, REF, ALT, sample_count, sample_ids
    "sample_ids"列包含空格分隔的样本ID

  SNP数据（宽格式）:
    CSV文件，列: sample_id, 1_3773191, 1_3052492, ...
    每行一个样本，每列一个SNP（0/1或缺失）

  环境数据:
    CSV文件，列: sample_id, PM2.5, PM10, SO2, NO2, CO, O3, AQI
    所有样本必须有环境数据

示例:
  python predict.py --snp my_snps.csv --env my_env.csv --out results.csv
  python predict.py --snp my_snps.csv --env my_env.csv --snp-format long --encoding gbk
  python predict.py --env my_env.csv --out results.csv  # 仅环境预测
        """
    )

    parser.add_argument('--snp', type=str, default=None,
                        help='SNP数据文件路径 (CSV格式)')
    parser.add_argument('--env', type=str, required=True,
                        help='环境数据文件路径 (CSV格式)')
    parser.add_argument('--out', type=str, default='predictions.csv',
                        help='输出文件路径 (默认: predictions.csv)')

    parser.add_argument('--snp-format', type=str, default='long',
                        choices=['long', 'wide'],
                        help='SNP数据格式: long=长格式, wide=宽格式')
    parser.add_argument('--snp-col', type=str, default='sample_ids',
                        help='长格式SNP数据中的样本列名 (默认: sample_ids)')
    parser.add_argument('--sample-col', type=str, default='sample_id',
                        help='宽格式/环境数据中的样本ID列名 (默认: sample_id)')
    parser.add_argument('--encoding', type=str, default=None,
                        help='文件编码，如 gbk, utf-8 (默认自动检测)')

    parser.add_argument('--model-dir', type=str, default=None,
                        help='模型文件所在目录 (默认: 工具包上级目录)')

    parser.add_argument('--env-only', action='store_true',
                        help='仅使用环境数据预测 (注意: 仅Spatial dissemination较可靠)')

    parser.add_argument('--interactive', action='store_true',
                        help='交互模式: 输入单个样本的环境数据')

    return parser.parse_args()


def interactive_mode(predictor):
    """Prompt for environment values and predict."""
    print("\n" + "="*60)
    print("4D 传播风险预测 - 交互模式")
    print("="*60)
    print("请输入环境参数 (单位与训练数据一致):")
    print("-"*60)

    values = []
    for feat in predictor.env_feature_cols:
        while True:
            try:
                val = input(f"  {feat}: ")
                values.append(float(val))
                break
            except ValueError:
                print("    请输入数字!")

    X_env = np.array([values], dtype=np.float32)

    print("\n" + "-"*60)
    print("预测中...")
    results = predictor.predict_env_only(X_env, sample_ids=['user_input'])

    print("\n" + "="*60)
    print("预测结果:")
    print("="*60)
    for dim in predictor.dim_names:
        score = results[dim][0]
        bar = "█" * int(score * 20)
        print(f"  {dim:<20} {score:.4f}  {bar}")

    print("\n  [注意] 此结果为仅环境数据预测，")
    print("         Transmission_Centrality / Clonal_Expansion / Persistence 仅供参考")
    print("="*60)


def main():
    args = parse_args()

    print("="*60)
    print("SVD(128) + ExtraTrees 4D 传播风险预测")
    print("="*60)

    try:
        predictor = TransmissionRiskPredictor(model_dir=args.model_dir)
    except Exception as e:
        print(f"[ERROR] 模型加载失败: {e}")
        sys.exit(1)

    if args.interactive:
        interactive_mode(predictor)
        return

    env_loader = EnvDataLoader()
    try:
        X_env, env_samples = env_loader.load(
            args.env,
            sample_col=args.sample_col,
            encoding=args.encoding
        )
    except Exception as e:
        print(f"[ERROR] 环境数据加载失败: {e}")
        sys.exit(1)

    if args.env_only or args.snp is None:
        results = predictor.predict_env_only(X_env, env_samples)
        save_results(results, args.out, env_only=True)
        return

    snp_loader = SNPDataLoader(predictor.snp_ids)
    try:
        if args.snp_format == 'long':
            X_snp, snp_samples = snp_loader.load_long_format(
                args.snp,
                samples_col=args.snp_col,
                encoding=args.encoding
            )
        else:
            X_snp, snp_samples = snp_loader.load_wide_format(
                args.snp,
                sample_col=args.sample_col,
                encoding=args.encoding
            )
    except Exception as e:
        print(f"[ERROR] SNP数据加载失败: {e}")
        sys.exit(1)

    try:
        X_snp_aligned, X_env_aligned, aligned_samples = align_samples(
            X_snp, snp_samples, X_env, env_samples
        )
    except Exception as e:
        print(f"[ERROR] 样本对齐失败: {e}")
        sys.exit(1)

    results = predictor.predict(X_snp_aligned, X_env_aligned, aligned_samples)
    save_results(results, args.out)


def save_results(results, out_path, env_only=False):
    """Save predictions to CSV."""
    df = pd.DataFrame(results)

    if '_note' in df.columns:
        df = df.drop(columns=['_note'])

    col_order = ['sample_id', 'Transmission_Centrality', 'Clonal_Expansion',
                 'Persistence', 'Spatial_Dissemination']
    df = df[[c for c in col_order if c in df.columns]]

    df.to_csv(out_path, index=False, encoding='utf-8-sig')

    print("\n" + "="*60)
    print(f"预测完成! 结果已保存: {out_path}")
    print("="*60)
    print(f"  样本数: {len(df)}")
    print(f"  预测维度: Transmission_Centrality, Clonal_Expansion, Persistence, Spatial_Dissemination")
    print()
    print("  统计摘要:")
    for col in ['Transmission_Centrality', 'Clonal_Expansion', 'Persistence', 'Spatial_Dissemination']:
        if col in df.columns:
            print(f"    {col:<20} min={df[col].min():.3f}  max={df[col].max():.3f}  mean={df[col].mean():.3f}")

    if env_only:
        print("\n  [WARNING] 此结果为仅环境数据预测")
        print("            Transmission_Centrality/Clonal_Expansion/Persistence 仅供参考")
    print("="*60)


if __name__ == '__main__':
    main()
