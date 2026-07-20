# Copyright (c) 2025 Peking University People's Hospital Hui Lab
# SPDX-License-Identifier: MIT
"""VCF parser for Snippy output."""
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import os
import gzip
import glob


class VCFParser:
    """Parse VCF files into a sparse SNP matrix aligned to reference SNP IDs."""

    def __init__(self, reference_snp_ids):
        self.reference_snp_ids = reference_snp_ids
        self.snp_to_idx = {snp_id: i for i, snp_id in enumerate(reference_snp_ids)}
        self.n_snps = len(reference_snp_ids)

    def _open(self, vcf_path):
        if vcf_path.endswith('.gz'):
            return gzip.open(vcf_path, 'rt')
        return open(vcf_path, 'r')

    def _has_alt(self, gt_val):
        """Return True if genotype contains a non-reference allele."""
        if gt_val in ('.', './.', '.|.'):
            return False
        for allele in gt_val.replace('|', '/').split('/'):
            if allele not in ('0', '.') and allele != '':
                return True
        return False

    def parse_multisample_vcf(self, vcf_path, min_quality=0):
        """Parse a multi-sample VCF and return the SNP matrix, sample names, and matched count."""
        print(f"[INFO] Parsing VCF: {vcf_path}")

        sample_names = []
        gt_start_col = -1

        with self._open(vcf_path) as f:
            for line in f:
                if line.startswith('##'):
                    continue
                if line.startswith('#CHROM'):
                    parts = line.strip().split('\t')
                    sample_names = parts[9:]
                    gt_start_col = 9
                    break

        print(f"       Samples found: {len(sample_names)}")
        print(f"       Reference SNPs: {self.n_snps}")

        if len(sample_names) == 0:
            raise ValueError("No samples found in VCF header")

        row_indices = []
        col_indices = []
        data = []
        matched_snps = set()
        total_vcf_snps = 0

        with self._open(vcf_path) as f:
            for line in f:
                if line.startswith('#'):
                    continue

                parts = line.strip().split('\t')
                if len(parts) < gt_start_col + 1:
                    continue

                chrom, pos, ref, alt, qual_str, fmt = parts[0], parts[1], parts[3], parts[4], parts[5], parts[8]

                if len(ref) != 1 or len(alt) != 1:
                    continue

                if qual_str != '.' and min_quality > 0:
                    try:
                        if float(qual_str) < min_quality:
                            continue
                    except ValueError:
                        pass

                snp_id = f"{chrom}_{pos}"
                if snp_id not in self.snp_to_idx:
                    continue

                total_vcf_snps += 1
                matched_snps.add(snp_id)
                ref_col_idx = self.snp_to_idx[snp_id]

                fmt_fields = fmt.split(':')
                gt_idx = fmt_fields.index('GT') if 'GT' in fmt_fields else 0

                for sample_idx, sample_gt in enumerate(parts[gt_start_col:]):
                    gt_val = sample_gt.split(':')[gt_idx] if ':' in sample_gt else sample_gt
                    if self._has_alt(gt_val):
                        row_indices.append(sample_idx)
                        col_indices.append(ref_col_idx)
                        data.append(1.0)

        X_sparse = csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(len(sample_names), self.n_snps),
            dtype=np.float32
        )

        print(f"       VCF SNPs (biallelic): {total_vcf_snps}")
        print(f"       Matched to reference: {len(matched_snps)}")
        print(f"       Sparse matrix: {X_sparse.shape}")
        print(f"       Non-zero entries: {X_sparse.nnz:,}")
        print(f"       Density: {X_sparse.nnz / (X_sparse.shape[0] * X_sparse.shape[1]) * 100:.4f}%")

        return X_sparse, sample_names, len(matched_snps)

    def parse_snippy_folder(self, folder_path):
        """Parse a folder with per-sample Snippy output subfolders."""
        print(f"[INFO] Parsing Snippy folder: {folder_path}")

        vcf_files = []
        for root, dirs, files in os.walk(folder_path):
            if 'snps.vcf' in files:
                sample_name = os.path.basename(root)
                vcf_files.append((sample_name, os.path.join(root, 'snps.vcf')))

        if not vcf_files:
            for vcf_path in glob.glob(os.path.join(folder_path, '*', '*.vcf')):
                sample_name = os.path.basename(os.path.dirname(vcf_path))
                vcf_files.append((sample_name, vcf_path))

        print(f"       Found {len(vcf_files)} sample VCFs")

        if not vcf_files:
            raise ValueError(f"No VCF files found in {folder_path}")

        row_indices = []
        col_indices = []
        data = []
        sample_names = []
        all_matched = set()

        for sample_idx, (sample_name, vcf_path) in enumerate(vcf_files):
            sample_names.append(sample_name)

            with self._open(vcf_path) as f:
                for line in f:
                    if line.startswith('#'):
                        continue

                    parts = line.strip().split('\t')
                    if len(parts) < 10:
                        continue

                    chrom, pos, ref, alt, fmt, gt = parts[0], parts[1], parts[3], parts[4], parts[8], parts[9]

                    if len(ref) != 1 or len(alt) != 1:
                        continue

                    snp_id = f"{chrom}_{pos}"
                    if snp_id not in self.snp_to_idx:
                        continue

                    all_matched.add(snp_id)
                    ref_col_idx = self.snp_to_idx[snp_id]

                    gt_val = gt.split(':')[0] if ':' in gt else gt
                    if self._has_alt(gt_val):
                        row_indices.append(sample_idx)
                        col_indices.append(ref_col_idx)
                        data.append(1.0)

        X_sparse = csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(len(sample_names), self.n_snps),
            dtype=np.float32
        )

        print(f"       Matched SNPs: {len(all_matched)}")
        print(f"       Sparse matrix: {X_sparse.shape}")
        print(f"       Non-zero entries: {X_sparse.nnz:,}")

        return X_sparse, sample_names, len(all_matched)


def vcf_to_model_matrix(vcf_path, model_snp_ids, output_csv=None):
    """Convert a VCF directly to a model-ready sparse matrix."""
    parser = VCFParser(model_snp_ids)
    X_sparse, sample_names, matched = parser.parse_multisample_vcf(vcf_path)

    if output_csv:
        pd.DataFrame({'sample_id': sample_names}).to_csv(output_csv, index=False)
        print(f"[INFO] Sample list saved: {output_csv}")

    return X_sparse, sample_names
