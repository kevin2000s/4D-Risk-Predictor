#!/usr/bin/env python3
"""
批量预测脚本
支持处理大量样本，自动分批次预测

用法:
    python batch_predict.py --snp-dir snp_data/ --env-file env.csv --out-dir results/
    python batch_predict.py --snp-file large_snp.csv --env-file env.csv --out results.csv --batch-size 100
"""
import argparse
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import TransmissionRiskPredictor
from data_loader import SNPDataLoader, EnvDataLoader, align_samples


def parse_args():
    parser = argparse.ArgumentParser(description='批量4D传播风险预测')

    parser.add_argument('--snp-file', type=str, default=None,
                        help='SNP数据文件（长格式或宽格式）')
    parser.add_argument('--snp-dir', type=str, default=None,
                        help='SNP数据目录（包含多个CSV文件）')
    parser.add_argument('--env-file', type=str, required=True,
                        help='环境数据文件')

    parser.add_argument('--out', type=str, default='batch_predictions.csv',
                        help='输出文件路径')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='输出目录（用于分文件输出）')

    parser.add_argument('--snp-format', type=str, default='long',
                        choices=['long', 'wide'])
    parser.add_argument('--batch-size', type=int, default=500,
                        help='每批处理的样本数（默认500）')
    parser.add_argument('--encoding', type=str, default=None)
    parser.add_argument('--model-dir', type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    print("="*60)
    print("批量 4D 传播风险预测")
    print("="*60)

    # 加载模型
    predictor = TransmissionRiskPredictor(model_dir=args.model_dir)

    # 加载环境数据
    env_loader = EnvDataLoader()
    X_env_all, env_samples_all = env_loader.load(
        args.env_file, encoding=args.encoding
    )

    all_results = []

    # 处理单个SNP文件
    if args.snp_file:
        print(f"\n处理SNP文件: {args.snp_file}")
        snp_loader = SNPDataLoader(predictor.snp_ids)

        if args.snp_format == 'long':
            X_snp, snp_samples = snp_loader.load_long_format(
                args.snp_file, encoding=args.encoding
            )
        else:
            X_snp, snp_samples = snp_loader.load_wide_format(
                args.snp_file, encoding=args.encoding
            )

        # 对齐并预测
        X_snp_a, X_env_a, samples = align_samples(
            X_snp, snp_samples, X_env_all, env_samples_all
        )

        # 分批预测
        n = len(samples)
        print(f"总样本数: {n}, 批次大小: {args.batch_size}")

        for start in range(0, n, args.batch_size):
            end = min(start + args.batch_size, n)
            print(f"  批次 {start//args.batch_size + 1}: 样本 {start}-{end}")

            batch_results = predictor.predict(
                X_snp_a[start:end],
                X_env_a[start:end],
                samples[start:end]
            )
            all_results.append(pd.DataFrame(batch_results))

    # 处理SNP目录
    elif args.snp_dir:
        import glob
        snp_files = sorted(glob.glob(os.path.join(args.snp_dir, "*.csv")))
        print(f"\n找到 {len(snp_files)} 个SNP文件")

        snp_loader = SNPDataLoader(predictor.snp_ids)

        for i, snp_file in enumerate(snp_files):
            print(f"\n[{i+1}/{len(snp_files)}] 处理: {os.path.basename(snp_file)}")

            if args.snp_format == 'long':
                X_snp, snp_samples = snp_loader.load_long_format(
                    snp_file, encoding=args.encoding
                )
            else:
                X_snp, snp_samples = snp_loader.load_wide_format(
                    snp_file, encoding=args.encoding
                )

            try:
                X_snp_a, X_env_a, samples = align_samples(
                    X_snp, snp_samples, X_env_all, env_samples_all
                )

                batch_results = predictor.predict(X_snp_a, X_env_a, samples)
                df_batch = pd.DataFrame(batch_results)
                all_results.append(df_batch)

                # 可选: 每个文件单独输出
                if args.out_dir:
                    os.makedirs(args.out_dir, exist_ok=True)
                    out_name = os.path.splitext(os.path.basename(snp_file))[0] + "_pred.csv"
                    out_path = os.path.join(args.out_dir, out_name)
                    df_batch.to_csv(out_path, index=False, encoding='utf-8-sig')
                    print(f"       已保存: {out_path}")

            except ValueError as e:
                print(f"       [SKIP] {e}")
                continue

    # 仅环境预测
    else:
        print("\n仅环境数据预测模式")
        results = predictor.predict_env_only(X_env_all, env_samples_all)
        all_results.append(pd.DataFrame(results))

    # 合并并保存
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)

        # 去重（保留第一个）
        final_df = final_df.drop_duplicates(subset=['sample_id'], keep='first')

        final_df.to_csv(args.out, index=False, encoding='utf-8-sig')

        print("\n" + "="*60)
        print(f"批量预测完成! 结果保存: {args.out}")
        print("="*60)
        print(f"  总样本数: {len(final_df)}")
        print(f"  预测维度: Network_Hub, Clone_Advantage, Persistence, Spatial_Connectivity")
        print()
        for col in ['Network_Hub', 'Clone_Advantage', 'Persistence', 'Spatial_Connectivity']:
            print(f"  {col:<20} min={final_df[col].min():.3f}  max={final_df[col].max():.3f}  mean={final_df[col].mean():.3f}")
        print("="*60)
    else:
        print("[WARNING] 没有成功预测任何样本")


if __name__ == '__main__':
    main()
