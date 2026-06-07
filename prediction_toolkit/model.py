"""
SVD(128) + ExtraTrees 4D Transmission Risk Prediction Model
Model wrapper for loading and inference.
"""
import joblib
import numpy as np
from scipy.sparse import csr_matrix
import os
import sys
import warnings
warnings.filterwarnings('ignore')


def _get_resource_dir():
    """
    获取资源文件所在目录。
    支持 PyInstaller 打包后的运行环境（sys._MEIPASS）和开发环境。
    """
    if getattr(sys, 'frozen', False):
        # PyInstaller 打包后的运行环境
        # sys.executable 指向 .exe 文件
        return os.path.dirname(sys.executable)
    else:
        # 开发环境：使用当前文件的上级目录
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TransmissionRiskPredictor:
    """
    4D传播风险预测器

    输入: SNP稀疏矩阵 (n_samples x 151,913) + 环境特征 (n_samples x 7)
    输出: 4维传播风险分数 (Network_Hub, Clone_Advantage, Persistence, Spatial_Connectivity)
    """

    def __init__(self, model_dir=None):
        """
        加载预训练模型

        Parameters
        ----------
        model_dir : str, optional
            模型文件所在目录。默认自动检测。
        """
        if model_dir is None:
            model_dir = _get_resource_dir()

        self.model_dir = model_dir
        self._load_models()

    def _resolve_model_path(self, filename):
        """解析模型文件路径，支持多种位置"""
        # 尝试路径优先级：model_dir > _internal > sys._MEIPASS
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

        # 返回第一个候选路径（用于错误提示）
        return os.path.join(candidates[0], filename)

    def _load_models(self):
        """加载所有模型组件"""
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

        # 构建SNP ID到索引的映射
        self.snp_to_idx = {snp_id: i for i, snp_id in enumerate(self.snp_ids)}

        print(f"[INFO] 模型加载完成:")
        print(f"       - SNP位点数: {len(self.snp_ids):,}")
        print(f"       - SVD维度: {self.svd.components_.shape[0]}")
        print(f"       - 环境特征: {self.env_feature_cols}")
        print(f"       - 预测维度: {self.dim_names}")

    def predict(self, X_snp_sparse, X_env, sample_ids=None):
        """
        对新样本进行4D传播风险预测

        Parameters
        ----------
        X_snp_sparse : scipy.sparse.csr_matrix
            稀疏SNP矩阵 (n_samples x 151,913)，列顺序必须与训练时的snp_ids一致
        X_env : np.ndarray or pd.DataFrame
            环境特征矩阵 (n_samples x 7)，顺序: [PM2.5, PM10, SO2, NO2, CO, O3, AQI]
        sample_ids : list, optional
            样本ID列表，用于输出标识

        Returns
        -------
        dict : {
            'sample_id': [...],
            'Network_Hub': [...],
            'Clone_Advantage': [...],
            'Persistence': [...],
            'Spatial_Connectivity': [...]
        }
        """
        n_samples = X_snp_sparse.shape[0]

        # 1. SVD降维
        print(f"[INFO] SVD降维: {X_snp_sparse.shape} -> ({n_samples}, {self.svd.components_.shape[0]})")
        X_svd = self.svd.transform(X_snp_sparse)

        # 2. 环境特征标准化
        print(f"[INFO] 环境特征标准化...")
        X_env_s = self.scaler.transform(X_env)

        # 3. 特征拼接
        X_full = np.hstack([X_svd, X_env_s])
        print(f"[INFO] 完整特征矩阵: {X_full.shape}")

        # 4. 预测4个维度
        print(f"[INFO] 预测4D传播风险...")
        results = {}
        if sample_ids is not None:
            results['sample_id'] = sample_ids
        else:
            results['sample_id'] = [f"sample_{i}" for i in range(n_samples)]

        for dim_name, model in self.models.items():
            preds = model.predict(X_full)
            results[dim_name] = preds
            print(f"       {dim_name}: range [{preds.min():.3f}, {preds.max():.3f}], mean={preds.mean():.3f}")

        return results

    def predict_env_only(self, X_env, sample_ids=None):
        """
        仅用环境数据预测（Spatial Connectivity维度相对可靠，其他维度仅供参考）

        Parameters
        ----------
        X_env : np.ndarray or pd.DataFrame
            环境特征矩阵 (n_samples x 7)
        sample_ids : list, optional
            样本ID列表

        Returns
        -------
        dict : 预测结果（含可信度标记）
        """
        n_samples = X_env.shape[0] if hasattr(X_env, 'shape') else len(X_env)

        print("[WARNING] 仅使用环境数据进行预测！")
        print("          Network_Hub / Clone_Advantage / Persistence 的预测可靠性较低")
        print("          Spatial_Connectivity 的预测相对可靠（~81%环境驱动）")

        # 使用零矩阵作为SNP输入（SVD将产生零向量）
        X_snp_zero = csr_matrix((n_samples, len(self.snp_ids)), dtype=np.float32)

        results = self.predict(X_snp_zero, X_env, sample_ids)
        results['_note'] = 'env_only_prediction'

        return results

    def get_feature_importance(self):
        """获取特征重要性（从训练结果）"""
        return {name: model.feature_importances_
                for name, model in self.models.items()}

    def explain_prediction(self, sample_idx=0):
        """
        解释单个样本的预测（返回各维度的Top驱动特征）

        Parameters
        ----------
        sample_idx : int
            样本索引

        Returns
        -------
        dict : 各维度的Top 5特征
        """
        # 这需要保存训练时的特征重要性
        # 从 feature_importance CSV 加载
        import pandas as pd

        fi_path = os.path.join(self.model_dir, 'svd128_extratrees_feature_importance.csv')
        if not os.path.exists(fi_path):
            return {"error": "特征重要性文件不存在"}

        fi_df = pd.read_csv(fi_path)
        fi_df = fi_df.sort_values('mean_importance', ascending=False)

        top_features = fi_df.head(10)[['feature', 'mean_importance', 'type']]
        return top_features.to_dict('records')
