from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchvision import models, transforms

IMG_SIZE = 224


def _build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


class ConvNeXtTransformerHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        nhead: int = 8,
        num_layers: int = 2,
    ):
        super().__init__()
        self.projection = nn.Linear(input_dim, hidden_dim)
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=0.2,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x).unsqueeze(1)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)


class ConvNeXtBaseTransformer(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        base_model = models.convnext_base(weights="IMAGENET1K_V1")
        self.cnn = nn.Sequential(*list(base_model.children())[:-2])
        self.transformer_head = ConvNeXtTransformerHead(
            input_dim=1024,
            hidden_dim=512,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = x.mean(dim=[2, 3])
        return self.transformer_head(x)


class ResNetTransformerHead(nn.Module):
    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 256,
        num_classes: int = 3,
        nhead: int = 8,
        num_layers: int = 2,
    ):
        super().__init__()
        self.projection = nn.Linear(input_dim, hidden_dim)
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x).unsqueeze(1)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.mean(dim=0)
        return self.classifier(x)


class CNNTransformerModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        base_model = models.resnet50(weights="IMAGENET1K_V2")
        self.cnn = nn.Sequential(*list(base_model.children())[:-1])
        self.head = ResNetTransformerHead(input_dim=2048, hidden_dim=256, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x).flatten(1)
        return self.head(x)


class EfficientNetB5Model(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        base_model = models.efficientnet_b5(weights="IMAGENET1K_V1")
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


@dataclass
class LoadedModel:
    name: str
    model: nn.Module


class FederatedMajorityEnsemble:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = self._resolve_classes()
        self.transform = _build_transform()
        self.models = self._load_models()

    def _resolve_classes(self) -> List[str]:
        train_dir = self.repo_root / "split_data" / "train"
        if train_dir.exists():
            classes = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
            if classes:
                return classes
        return ["covid", "normal", "pneumonia"]

    def _load_state_dict(self, model: nn.Module, checkpoint_path: Path) -> nn.Module:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        state = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        return model

    def _load_models(self) -> List[LoadedModel]:
        num_classes = len(self.classes)
        convnext = self._load_state_dict(
            ConvNeXtBaseTransformer(num_classes=num_classes),
            self.repo_root / "convnext-models" / "best_convnext_base_model.pth",
        )
        resnet = self._load_state_dict(
            CNNTransformerModel(num_classes=num_classes),
            self.repo_root / "resnet-models" / "best_model_resnet50_gradaccum.pth",
        )
        efficientnet = self._load_state_dict(
            EfficientNetB5Model(num_classes=num_classes),
            self.repo_root / "efficient-net-models" / "best_model_efficientnet_only_CNN.pth",
        )

        return [
            LoadedModel(name="convnext_base", model=convnext),
            LoadedModel(name="resnet50_transformer", model=resnet),
            LoadedModel(name="efficientnet_b5_cnn", model=efficientnet),
        ]

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> Dict[str, object]:
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

        per_model: List[Dict[str, object]] = []
        vote_counts = {c: 0 for c in self.classes}
        avg_probs = torch.zeros(len(self.classes), device=self.device)

        for loaded in self.models:
            logits = loaded.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            avg_probs += probs

            pred_idx = int(torch.argmax(probs).item())
            pred_class = self.classes[pred_idx]
            vote_counts[pred_class] += 1

            per_model.append(
                {
                    "model": loaded.name,
                    "predicted_class": pred_class,
                    "confidence": float(probs[pred_idx].item()),
                }
            )

        avg_probs /= len(self.models)

        winner, votes = max(vote_counts.items(), key=lambda item: item[1])
        tied_labels = [label for label, count in vote_counts.items() if count == votes]

        if len(tied_labels) > 1:
            tied_indices = [self.classes.index(label) for label in tied_labels]
            tie_winner_idx = max(tied_indices, key=lambda idx: float(avg_probs[idx].item()))
            winner = self.classes[tie_winner_idx]

        winner_idx = self.classes.index(winner)

        return {
            "final_class": winner,
            "final_confidence": float(avg_probs[winner_idx].item()),
            "votes": vote_counts,
            "per_model": per_model,
            "class_probabilities": {
                class_name: float(avg_probs[idx].item())
                for idx, class_name in enumerate(self.classes)
            },
        }
