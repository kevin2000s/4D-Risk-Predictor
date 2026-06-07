"""
VCF Parser for Snippy Output
Supports multi-sample VCF files and converts to SNP matrix compatible with the model.
"""
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import os
import gzip


class VCFParser:
    """
    Parse Snippy VCF output and convert to model-compatible SNP sparse matrix.

    The VCF should contain GT (genotype) format with 0=reference, 1=alternate.
    Missing SNPs (not in VCF) are filled with 0 (assumed reference).
    """

    def __init__(self, reference_snp_ids):
        """
        Parameters
        ----------
        reference_snp_ids : list
            Training SNP IDs in format "CHROM_POS", e.g., ["1_3773191", "1_3052492", ...]
        """
        self.reference_snp_ids = reference_snp_ids
        self.snp_to_idx = {snp_id: i for i, snp_id in enumerate(reference_snp_ids)}
        self.n_snps = len(reference_snp_ids)

    def parse_multisample_vcf(self, vcf_path, min_quality=0):
        """
        Parse a multi-sample VCF file (like snippy's raw_snps.vcf).

        Parameters
        ----------
        vcf_path : str
            Path to VCF file (.vcf or .vcf.gz)
        min_quality : int
            Minimum QUAL score to include a SNP (0 = include all)

        Returns
        -------
        X_sparse : scipy.sparse.csr_matrix
            Sparse matrix (n_samples x n_snps)
        sample_names : list
            List of sample names from VCF header
        matched_count : int
            Number of SNPs matched to reference
        """
        print(f"[INFO] Parsing VCF: {vcf_path}")

        # Determine if gzipped
        opener = gzip.open if vcf_path.endswith('.gz') else open
        mode = 'rt' if vcf_path.endswith('.gz') else 'r'

        sample_names = []
        chrom_col = -1
        pos_col = -1
        ref_col = -1
        alt_col = -1
        qual_col = -1
        format_col = -1
        gt_start_col = -1

        # Parse header
        with opener(vcf_path, mode) as f:
            for line in f:
                if line.startswith('##'):
                    continue
                if line.startswith('#CHROM'):
                    parts = line.strip().split('\t')
                    sample_names = parts[9:]
                    chrom_col = 0
                    pos_col = 1
                    ref_col = 3
                    alt_col = 4
                    qual_col = 5
                    format_col = 8
                    gt_start_col = 9
                    break

        print(f"       Samples found: {len(sample_names)}")
        print(f"       Reference SNPs: {self.n_snps}")

        if len(sample_names) == 0:
            raise ValueError("No samples found in VCF header")

        # Build sparse matrix data
        # For efficiency, use COO format then convert to CSR
        row_indices = []
        col_indices = []
        data = []

        matched_snps = set()
        total_vcf_snps = 0

        with opener(vcf_path, mode) as f:
            for line in f:
                if line.startswith('#'):
                    continue

                parts = line.strip().split('\t')
                if len(parts) < gt_start_col + 1:
                    continue

                chrom = parts[chrom_col]
                pos = parts[pos_col]
                ref = parts[ref_col]
                alt = parts[alt_col]
                qual_str = parts[qual_col]
                fmt = parts[format_col]

                # Skip indels if needed (keep only SNPs)
                if len(ref) != 1 or len(alt) != 1:
                    continue

                # Skip low quality
                if qual_str != '.' and min_quality > 0:
                    try:
                        if float(qual_str) < min_quality:
                            continue
                    except:
                        pass

                # Build SNP ID
                snp_id = f"{chrom}_{pos}"

                if snp_id not in self.snp_to_idx:
                    continue

                total_vcf_snps += 1
                matched_snps.add(snp_id)
                ref_col_idx = self.snp_to_idx[snp_id]

                # Parse GT field for each sample
                fmt_fields = fmt.split(':')
                gt_idx = fmt_fields.index('GT') if 'GT' in fmt_fields else 0

                for sample_idx, sample_gt in enumerate(parts[gt_start_col:]):
                    gt_val = sample_gt.split(':')[gt_idx] if ':' in sample_gt else sample_gt

                    # Parse genotype: 0=ref, 1=alt, ./.=missing
                    # Handle diploid: "0/1", "1/1", "0/0", "./."
                    if gt_val in ('.', './.', '.|.'):
                        continue

                    # Check if any alternate allele present
                    has_alt = False
                    for allele in gt_val.replace('|', '/').split('/'):
                        if allele not in ('0', '.') and allele != '':
                            has_alt = True
                            break

                    if has_alt:
                        row_indices.append(sample_idx)
                        col_indices.append(ref_col_idx)
                        data.append(1.0)

        # Build sparse matrix
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
        """
        Parse a folder containing per-sample Snippy output subfolders.
        Each subfolder should contain a snps.vcf file.

        Parameters
        ----------
        folder_path : str
            Path to folder containing sample subfolders

        Returns
        -------
        X_sparse : scipy.sparse.csr_matrix
            Sparse matrix (n_samples x n_snps)
        sample_names : list
            List of sample names (subfolder names)
        matched_count : int
            Number of SNPs matched to reference
        """
        print(f"[INFO] Parsing Snippy folder: {folder_path}")

        import glob

        # Find all snps.vcf files
        vcf_files = []
        for root, dirs, files in os.walk(folder_path):
            if 'snps.vcf' in files:
                sample_name = os.path.basename(root)
                vcf_files.append((sample_name, os.path.join(root, 'snps.vcf')))

        if not vcf_files:
            # Try alternative naming
            vcf_pattern = os.path.join(folder_path, '*', '*.vcf')
            for vcf_path in glob.glob(vcf_pattern):
                sample_name = os.path.basename(os.path.dirname(vcf_path))
                vcf_files.append((sample_name, vcf_path))

        print(f"       Found {len(vcf_files)} sample VCFs")

        if not vcf_files:
            raise ValueError(f"No VCF files found in {folder_path}")

        # Parse each VCF and merge
        row_indices = []
        col_indices = []
        data = []
        sample_names = []
        all_matched = set()

        for sample_idx, (sample_name, vcf_path) in enumerate(vcf_files):
            sample_names.append(sample_name)

            opener = gzip.open if vcf_path.endswith('.gz') else open
            mode = 'rt' if vcf_path.endswith('.gz') else 'r'

            with opener(vcf_path, mode) as f:
                for line in f:
                    if line.startswith('#'):
                        continue

                    parts = line.strip().split('\t')
                    if len(parts) < 10:
                        continue

                    chrom = parts[0]
                    pos = parts[1]
                    ref = parts[3]
                    alt = parts[4]
                    fmt = parts[8]
                    gt = parts[9]

                    # Skip indels
                    if len(ref) != 1 or len(alt) != 1:
                        continue

                    snp_id = f"{chrom}_{pos}"
                    if snp_id not in self.snp_to_idx:
                        continue

                    all_matched.add(snp_id)
                    ref_col_idx = self.snp_to_idx[snp_id]

                    # Parse GT
                    gt_val = gt.split(':')[0] if ':' in gt else gt
                    if gt_val in ('.', './.', '.|.'):
                        continue

                    has_alt = False
                    for allele in gt_val.replace('|', '/').split('/'):
                        if allele not in ('0', '.') and allele != '':
                            has_alt = True
                            break

                    if has_alt:
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
    """
    Convenience function: Convert VCF directly to model-ready sparse matrix.

    Parameters
    ----------
    vcf_path : str
        Path to VCF file
    model_snp_ids : list
        SNP IDs from the trained model
    output_csv : str, optional
        If provided, save sample_id list to CSV

    Returns
    -------
    X_sparse, sample_names
    """
    parser = VCFParser(model_snp_ids)
    X_sparse, sample_names, matched = parser.parse_multisample_vcf(vcf_path)

    if output_csv:
        pd.DataFrame({'sample_id': sample_names}).to_csv(output_csv, index=False)
        print(f"[INFO] Sample list saved: {output_csv}")

    return X_sparse, sample_names
