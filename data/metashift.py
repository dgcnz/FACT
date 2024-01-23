import lightning as L
import clip
from PIL import Image
from enum import Enum
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import torch
from torch.utils.data import DataLoader
import datasets
from concepts.concept_utils import ConceptBank


class NNProjector(torch.nn.Module):
    def __init__(self, concept_bank_path: str, backbone: torch.nn.Module):
        super().__init__()
        concept_bank = ConceptBank.from_pickle(
            pickle_path=concept_bank_path, device="cpu"
        )
        self.cb_vectors = concept_bank.concept_info.vectors
        self.cb_intercepts = concept_bank.concept_info.intercepts
        self.cb_norms = concept_bank.concept_info.norms
        self.concept_names = concept_bank.concept_names
        self.backbone = backbone

    def __call__(self, imgs: torch.Tensor):
        x = self.backbone(imgs)
        concept_weights = ((self.cb_vectors @ x.T) + self.cb_intercepts) / (
            self.cb_norms
        )
        return concept_weights.T


class CLIPPreprocessor(torch.nn.Module):
    def __init__(self, clip_model_name: str):
        super().__init__()
        _, self.preprocess = clip.load(clip_model_name, jit=True)

    def __call__(self, x: list[Image.Image]) -> torch.Tensor:
        with torch.no_grad():
            return torch.stack([self.preprocess(img) for img in x])


class PreprocessorEnum(Enum):
    CLIP_RESNET50 = 1
    RESNET18_IMAGENET = 2


PREPROCESSORS: dict[PreprocessorEnum, callable] = {
    PreprocessorEnum.CLIP_RESNET50: CLIPPreprocessor("RN50"),
}


class MetaShiftDataModule(L.LightningDataModule):
    def __init__(
        self,
        task_name: str,
        projector: NNProjector,
        preprocessor_name: PreprocessorEnum,
        train_batch_size: int = 32,
        test_batch_size: int = 32,
    ):
        super().__init__()
        self.task_name = task_name
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.projector = projector
        self.preprocessor = PREPROCESSORS[preprocessor_name]

    def setup(self, stage: str):
        self.dataset = datasets.load_dataset(
            "fact-40/pcbm_metashift", name=self.task_name, trust_remote_code=True
        )
        self.classes = self.dataset["train"].info.features["label"].names
        # train requires train and test
        # test only requires test
        if stage != "test":
            self.dataset["train"] = (
                self.dataset["train"]
                .map(self.convert_to_features, batched=True, remove_columns="image")
                .with_format("torch")
            )
        self.dataset["test"] = (
            self.dataset["test"]
            .map(self.convert_to_features, batched=True, remove_columns="image")
            .with_format("torch")
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"], batch_size=self.train_batch_size, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(self.dataset["test"], batch_size=self.test_batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset["test"], batch_size=self.test_batch_size)

    def convert_to_features(self, example_batch, indices=None):
        image = example_batch["image"]
        label = example_batch["label"]
        image = self.preprocessor(image)
        concept_weights = self.projector(image)
        assert isinstance(concept_weights, torch.Tensor), type(concept_weights)
        return {"concept_weights": concept_weights, "label": label}


def collate_fn(self):
    pass
