"""
数据加载和预处理模块
支持多种SNP数据格式输入：
  - 长格式CSV (类似 snp_sample_count.csv)
  - 宽格式CSV (样本 x SNP 矩阵)
  - VCF文件 (Snippy多样本VCF输出)
  - Snippy结果文件夹
"""
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import os
import re


class SNPDataLoader:
    """
    SNP数据加载器
    支持多种输入格式，输出标准稀疏矩阵
    """

    def __init__(self, reference_snp_ids):
        """
        Parameters
        ----------
        reference_snp_ids : list
            训练时的151,913个SNP ID列表，格式: "CHROM_POS"
        """
        self.reference_snp_ids = reference_snp_ids
        self.snp_to_idx = {snp_id: i for i, snp_id in enumerate(reference_snp_ids)}
        self.n_snps = len(reference_snp_ids)

    def load_long_format(self, filepath, chrom_col='CHROM', pos_col='POS',
                         samples_col='sample_ids', encoding=None):
        """
        加载长格式SNP数据（类似 snp_sample_count.csv）

        每行一个SNP，包含该SNP出现的所有样本ID（空格分隔）

        Parameters
        ----------
        filepath : str
            CSV文件路径
        chrom_col, pos_col : str
            染色体和位置列名
        samples_col : str
            样本列名（包含空格分隔的样本ID）
        encoding : str, optional
            文件编码，如 'gbk', 'utf-8'

        Returns
        -------
        scipy.sparse.csr_matrix, list
            稀疏矩阵 (n_samples x n_snps), 样本ID列表
        """
        print(f"[INFO] 加载长格式SNP数据: {filepath}")

        if encoding:
            df = pd.read_csv(filepath, encoding=encoding)
        else:
            try:
                df = pd.read_csv(filepath, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(filepath, encoding='gbk')

        # 创建SNP ID
        df['snp_id'] = df[chrom_col].astype(str) + '_' + df[pos_col].astype(str)

        # 去重（保留第一个）
        df = df.drop_duplicates(subset=['snp_id'], keep='first')
        print(f"       总SNP变异数: {len(df)}")

        # 收集所有样本
        all_samples = set()
        for s in df[samples_col].dropna():
            all_samples.update(str(s).strip().split())
        all_samples = sorted(list(all_samples))
        sample_to_idx = {s: i for i, s in enumerate(all_samples)}
        print(f"       唯一样本数: {len(all_samples)}")

        # 构建稀疏矩阵
        row_indices = []
        col_indices = []

        for col_idx, (_, row) in enumerate(df.iterrows()):
            snp_id = row['snp_id']
            if snp_id not in self.snp_to_idx:
                continue  # 跳过训练时没有的SNP

            ref_col_idx = self.snp_to_idx[snp_id]
            for sample in str(row[samples_col]).strip().split():
                if sample in sample_to_idx:
                    row_indices.append(sample_to_idx[sample])
                    col_indices.append(ref_col_idx)

        data = np.ones(len(row_indices), dtype=np.float32)
        X_sparse = csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(len(all_samples), self.n_snps),
            dtype=np.float32
        )

        print(f"       稀疏矩阵: {X_sparse.shape}")
        print(f"       非零项: {X_sparse.nnz:,} (密度: {X_sparse.nnz / (X_sparse.shape[0] * X_sparse.shape[1]) * 100:.4f}%)")

        return X_sparse, all_samples

    def load_wide_format(self, filepath, sample_col='sample_id', encoding=None):
        """
        加载宽格式SNP数据（样本 x SNP 矩阵）

        Parameters
        ----------
        filepath : str
            CSV文件路径
        sample_col : str
            样本ID列名
        encoding : str, optional
            文件编码

        Returns
        -------
        scipy.sparse.csr_matrix, list
            稀疏矩阵, 样本ID列表
        """
        print(f"[INFO] 加载宽格式SNP数据: {filepath}")

        if encoding:
            df = pd.read_csv(filepath, encoding=encoding)
        else:
            try:
                df = pd.read_csv(filepath, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(filepath, encoding='gbk')

        sample_ids = df[sample_col].astype(str).tolist()
        df = df.drop(columns=[sample_col])

        print(f"       样本数: {len(sample_ids)}")
        print(f"       SNP列数: {len(df.columns)}")

        # 列名格式应为 "CHROM_POS"
        valid_cols = [c for c in df.columns if c in self.snp_to_idx]
        print(f"       匹配参考SNP的列数: {len(valid_cols)}")

        # 重排列以匹配参考顺序
        aligned = pd.DataFrame(index=df.index, columns=self.reference_snp_ids)
        aligned[valid_cols] = df[valid_cols]
        aligned = aligned.fillna(0).astype(np.float32)

        X_sparse = csr_matrix(aligned.values)

        print(f"       稀疏矩阵: {X_sparse.shape}")
        print(f"       非零项: {X_sparse.nnz:,}")

        return X_sparse, sample_ids

    def load_vcf_list(self, vcf_files, sample_names=None):
        """
        从VCF文件列表构建SNP矩阵（简化版，提取 CHROM_POS 作为SNP ID）

        Parameters
        ----------
        vcf_files : list
            VCF文件路径列表
        sample_names : list, optional
            样本名称列表。默认使用文件名

        Returns
        -------
        scipy.sparse.csr_matrix, list
            稀疏矩阵, 样本ID列表
        """
        print(f"[INFO] 从VCF文件加载SNP数据 ({len(vcf_files)}个文件)")

        if sample_names is None:
            sample_names = [os.path.splitext(os.path.basename(f))[0] for f in vcf_files]

        sample_to_idx = {s: i for i, s in enumerate(sample_names)}

        row_indices = []
        col_indices = []

        for sample_idx, vcf_path in enumerate(vcf_files):
            snp_set = self._parse_vcf(vcf_path)
            for snp_id in snp_set:
                if snp_id in self.snp_to_idx:
                    row_indices.append(sample_idx)
                    col_indices.append(self.snp_to_idx[snp_id])

        data = np.ones(len(row_indices), dtype=np.float32)
        X_sparse = csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(len(sample_names), self.n_snps),
            dtype=np.float32
        )

        print(f"       稀疏矩阵: {X_sparse.shape}")
        print(f"       非零项: {X_sparse.nnz:,}")

        return X_sparse, sample_names

    def _parse_vcf(self, vcf_path):
        """解析VCF文件，返回SNP ID集合"""
        snp_set = set()
        with open(vcf_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    chrom = parts[0]
                    pos = parts[1]
                    snp_id = f"{chrom}_{pos}"
                    snp_set.add(snp_id)
        return snp_set


class EnvDataLoader:
    """环境数据加载器"""

    REQUIRED_COLS = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'AQI']

    def __init__(self):
        pass

    def load(self, filepath, sample_col='sample_id', encoding=None):
        """
        加载环境数据

        Parameters
        ----------
        filepath : str
            CSV文件路径
        sample_col : str
            样本ID列名
        encoding : str, optional
            文件编码

        Returns
        -------
        np.ndarray, list
            环境特征矩阵 (n_samples x 7), 样本ID列表
        """
        print(f"[INFO] 加载环境数据: {filepath}")

        if encoding:
            df = pd.read_csv(filepath, encoding=encoding)
        else:
            try:
                df = pd.read_csv(filepath, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(filepath, encoding='gbk')

        # 检查必要列
        missing = [c for c in self.REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"环境数据缺少列: {missing}。需要: {self.REQUIRED_COLS}")

        sample_ids = df[sample_col].astype(str).tolist()

        # 提取环境特征
        X_env = df[self.REQUIRED_COLS].copy()

        # 处理缺失值（用均值填充）
        for col in self.REQUIRED_COLS:
            if X_env[col].isna().any():
                mean_val = X_env[col].mean()
                X_env[col] = X_env[col].fillna(mean_val)
                print(f"       [WARN] {col} 有缺失值，已用均值 {mean_val:.2f} 填充")

        print(f"       样本数: {len(sample_ids)}")
        print(f"       特征: {self.REQUIRED_COLS}")

        return X_env.values.astype(np.float32), sample_ids


def align_samples(X_snp, snp_samples, X_env, env_samples):
    """
    对齐SNP样本和环境样本，取交集

    Returns
    -------
    X_snp_aligned, X_env_aligned, aligned_samples
    """
    common = sorted(list(set(snp_samples) & set(env_samples)))

    if len(common) == 0:
        raise ValueError("SNP样本和环境样本没有交集！请检查样本ID是否一致。")

    if len(common) < len(snp_samples) or len(common) < len(env_samples):
        print(f"[WARN] 样本对齐: SNP={len(snp_samples)}, 环境={len(env_samples)}, 交集={len(common)}")
        print(f"       仅对 {len(common)} 个共有样本进行预测")

    snp_idx = {s: i for i, s in enumerate(snp_samples)}
    env_idx = {s: i for i, s in enumerate(env_samples)}

    snp_indices = [snp_idx[s] for s in common]
    env_indices = [env_idx[s] for s in common]

    return X_snp[snp_indices, :], X_env[env_indices, :], common
