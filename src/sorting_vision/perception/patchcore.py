"""表面缺陷检测：PatchCore 无监督异常检测（torch，可选）+ 统计兜底（numpy）。

PatchCore 只用正常样本训练：提取预训练 ResNet 中间层特征做 memory bank，
测试时与 memory 最近邻距离作为异常分数，像素级分数上采样到原图再阈值化。
无 torch 时回退到统计检测（正常样本均值/标准差 + z-score）。
"""
from __future__ import annotations

import numpy as np


class StatisticalAnomalyDetector:
    """统计异常检测兜底：按像素 z-score 偏离正常样本的程度判异常。"""

    def __init__(self, n_sigma: float = 3.0):
        self.n_sigma = n_sigma
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, normal_images: list[np.ndarray]) -> None:
        stack = np.stack([im.astype(np.float32) for im in normal_images])
        self.mean = stack.mean(axis=0)
        self.std = stack.std(axis=0) + 1e-6

    def predict(self, image: np.ndarray) -> dict:
        if self.mean is None:
            raise RuntimeError("需先 fit 正常样本")
        z = np.abs(image.astype(np.float32) - self.mean) / self.std
        mask = (z > self.n_sigma).astype(np.uint8) * 255
        score = float(z.mean())
        return {
            "anomaly_score": score,
            "is_defect": bool(score > self.n_sigma * 0.35),
            "threshold": float(self.n_sigma),
            "mask": mask,
            "score_map": z,
        }


class PatchCoreDetector:
    """PatchCore（torch）：ResNet 中间层特征 + memory bank + 最近邻打分。"""

    def __init__(
        self,
        layers: tuple[str, ...] = ("layer2", "layer3"),
        feature_size: int = 32,
        coreset_ratio: float = 0.1,
        n_sigma: float = 3.0,
        device: str = "cpu",
    ):
        import torch
        import torchvision

        self.torch = torch
        self.feature_size = feature_size
        self.coreset_ratio = coreset_ratio
        self.n_sigma = n_sigma
        self.device = device

        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        self.model = torchvision.models.resnet18(weights=weights)
        self.model.eval().to(device)
        self._feats: dict[str, "torch.Tensor"] = {}
        self._hooks = []
        named = dict(self.model.named_modules())
        for name in layers:
            self._hooks.append(named[name].register_forward_hook(self._make_hook(name)))
        self.layers = layers
        self.memory: "torch.Tensor | None" = None
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def _make_hook(self, name: str):
        def hook(module, inp, out):
            self._feats[name] = out.detach()

        return hook

    def _to_tensor(self, images: list[np.ndarray]):
        batch = []
        for im in images:
            if im.ndim == 2:
                im = np.stack([im] * 3, axis=-1)
            im = im.astype(np.float32) / 255.0
            im = self.torch.from_numpy(im).permute(2, 0, 1)
            im = self.normalize(im)
            batch.append(im)
        return self.torch.stack(batch).to(self.device)

    def _extract(self, x):
        import torch.nn.functional as F

        self._feats = {}
        with self.torch.no_grad():
            self.model(x)
        feats = []
        for name in self.layers:
            f = self._feats[name]  # (N, C, h, w)
            f = F.adaptive_avg_pool2d(f, (self.feature_size, self.feature_size))
            feats.append(f)
        fused = self.torch.cat(feats, dim=1)  # (N, C_total, S, S)
        return fused.flatten(2).transpose(1, 2)  # (N, P, C)

    def fit(self, normal_images: list[np.ndarray]) -> None:
        x = self._to_tensor(normal_images)
        feats = self._extract(x)  # (N, P, C)
        bank = feats.reshape(-1, feats.shape[-1])  # (N*P, C)
        k = max(1, int(len(bank) * self.coreset_ratio))
        idx = self.torch.randperm(len(bank), device=self.device)[:k]
        self.memory = bank[idx]
        # 用正常样本的最近邻分数校准阈值（max + 20% 裕量）
        with self.torch.no_grad():
            d = self.torch.cdist(feats, self.memory)  # (N, P, M)
            normal_scores = d.min(dim=2).values.amax(dim=1)  # (N,)
        self.threshold = float(normal_scores.max().item()) * 1.2

    def predict(self, image: np.ndarray) -> dict:
        if self.memory is None:
            raise RuntimeError("需先 fit 正常样本")
        x = self._to_tensor([image])
        feats = self._extract(x)[0]  # (P, C)
        with self.torch.no_grad():
            d = self.torch.cdist(feats, self.memory)  # (P, M)
            score = d.min(dim=1).values  # (P,)
        score_map = score.view(self.feature_size, self.feature_size).cpu().numpy()

        import cv2

        h, w = image.shape[:2]
        score_map = cv2.resize(score_map, (w, h), interpolation=cv2.INTER_LINEAR)
        score_map = cv2.GaussianBlur(score_map, (5, 5), 0)
        mask = (score_map > self.threshold).astype(np.uint8) * 255
        return {
            "anomaly_score": float(score_map.max()),
            "is_defect": bool(float(score_map.max()) > self.threshold),
            "threshold": self.threshold,
            "mask": mask,
            "score_map": score_map,
        }
