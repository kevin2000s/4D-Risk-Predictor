# Copyright (c) 2025 Peking University People's Hospital Hui Lab
# SPDX-License-Identifier: MIT
"""SVD(128) + ExtraTrees 4D transmission risk model wrapper."""
import joblib
import numpy as np
from scipy.sparse import csr_matrix
import os
import sys
import warnings

warnings.filterwarnings('ignore')


def _get_resource_dir():
    """Return the directory containing model files."""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TransmissionRiskPredictor:
    """Load pre-trained models and predict 4D transmission risk scores."""

    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = _get_resource_dir()
        self.model_dir = model_dir
        self._load_models()

    def _resolve_model_path(self, filename):
        """Locate a model file in the packaged or development layout."""
        candidates = [self.model_dir]
        if getattr(sys, 'frozen', False):
            candidates.append(os.path.join(self.model_dir, '_internal'))
            if hasattr(sys, '_MEIPASS'):
                candidates.append(sys._MEIPASS)
                candidates.append(os.path.join(sys._MEIPASS, '_internal'))

        for base in candidates:
            path = os.path.join(base, filename)
            if os.path.exists(path):
                return path
        return os.path.join(candidates[0], filename)

    def _load_models(self):
        """Load model components from joblib files."""
        print("[INFO] 加载模型组件...")

        paths = {
            'models': self._resolve_model_path('svd128_extratrees_models.joblib'),
            'svd': self._resolve_model_path('svd128_svd_transformer.joblib'),
            'scaler': self._resolve_model_path('svd128_env_scaler.joblib'),
            'metadata': self._resolve_model_path('svd128_model_metadata.joblib'),
        }

        for name, path in paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"模型文件缺失: {path}")

        self.models = joblib.load(paths['models'])
        self.svd = joblib.load(paths['svd'])
        self.scaler = joblib.load(paths['scaler'])
        self.metadata = joblib.load(paths['metadata'])

        self.snp_ids = self.metadata['snp_ids']
        self.env_feature_cols = self.metadata['env_feature_cols']
        self.dim_names = self.metadata['dim_names']
        self.feature_names = self.metadata['feature_names']
        self.snp_to_idx = {snp_id: i for i, snp_id in enumerate(self.snp_ids)}

        print("[INFO] 模型加载完成:")
        print(f"       - SNP位点数: {len(self.snp_ids):,}")
        print(f"       - SVD维度: {self.svd.components_.shape[0]}")
        print(f"       - 环境特征: {self.env_feature_cols}")
        print(f"       - 预测维度: {self.dim_names}")

    def predict(self, X_snp_sparse, X_env, sample_ids=None):
        """Predict 4D risk scores from SNP and environment data.

        Returns a dict with sample_id and one key per dimension.
        """
        n_samples = X_snp_sparse.shape[0]

        print(f"[INFO] SVD降维: {X_snp_sparse.shape} -> ({n_samples}, {self.svd.components_.shape[0]})")
        X_svd = self.svd.transform(X_snp_sparse)

        print("[INFO] 环境特征标准化...")
        X_env_s = self.scaler.transform(X_env)

        X_full = np.hstack([X_svd, X_env_s])
        print(f"[INFO] 完整特征矩阵: {X_full.shape}")

        print("[INFO] 预测4D传播风险...")
        results = {'sample_id': sample_ids if sample_ids is not None else [f"sample_{i}" for i in range(n_samples)]}

        for dim_name, model in self.models.items():
            preds = model.predict(X_full)
            results[dim_name] = preds
            print(f"       {dim_name}: range [{preds.min():.3f}, {preds.max():.3f}], mean={preds.mean():.3f}")

        return results

    def predict_env_only(self, X_env, sample_ids=None):
        """Predict using environment data only.

        Spatial dissemination is mainly environment-driven; the other
        dimensions are less reliable without SNP data.
        """
        n_samples = X_env.shape[0] if hasattr(X_env, 'shape') else len(X_env)
        X_snp_zero = csr_matrix((n_samples, len(self.snp_ids)), dtype=np.float32)

        print("[WARNING] 仅使用环境数据进行预测！")
        print("          Transmission_Centrality / Clonal_Expansion / Persistence 的预测可靠性较低")
        print("          Spatial_Dissemination 的预测相对可靠（~81%环境驱动）")

        results = self.predict(X_snp_zero, X_env, sample_ids)
        results['_note'] = 'env_only_prediction'
        return results

    def get_feature_importance(self):
        """Return per-dimension ExtraTrees feature importances."""
        return {name: model.feature_importances_
                for name, model in self.models.items()}

    def explain_prediction(self, sample_idx=0):
        """Return the top 10 globally important features."""
        import pandas as pd

        fi_path = os.path.join(self.model_dir, 'svd128_extratrees_feature_importance.csv')
        if not os.path.exists(fi_path):
            return {"error": "特征重要性文件不存在"}

        fi_df = pd.read_csv(fi_path)
        fi_df = fi_df.sort_values('mean_importance', ascending=False)
        top_features = fi_df.head(10)[['feature', 'mean_importance', 'type']]
        return top_features.to_dict('records')
