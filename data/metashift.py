import lightning as L
from torchvision import transforms
import logging
import clip
from PIL import Image
from enum import Enum
import torch
from torch.utils.data import DataLoader
import datasets
from pathlib import Path
from concepts.concept_utils import ConceptBank


class NNProjector(torch.nn.Module):
    name: str

    def __init__(self, concept_bank_path: str, backbone: torch.nn.Module):
        super().__init__()
        self.name = Path(concept_bank_path).stem
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
    def __init__(self, clip_model_name: str, device: str = "cpu"):
        super().__init__()
        _, self.preprocess = clip.load(clip_model_name, jit=True, device=device)

    def __call__(self, x: list[Image.Image]) -> torch.Tensor:
        with torch.no_grad():
            return torch.stack([self.preprocess(img) for img in x])


class ImageNetPreprocessor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transforms = transforms.Compose(
            [
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def forward(self, x: list[Image.Image]) -> torch.Tensor:
        return torch.stack([self.transforms(img.convert("RGB")) for img in x])


class PreprocessorEnum(Enum):
    CLIP_RESNET50 = 1
    RESNET18_IMAGENET_1K_V1 = 2


PREPROCESSORS: dict[PreprocessorEnum, callable] = {
    PreprocessorEnum.CLIP_RESNET50: CLIPPreprocessor("RN50"),
    PreprocessorEnum.RESNET18_IMAGENET_1K_V1: ImageNetPreprocessor(),
}


class MetaShiftDataModule(L.LightningDataModule):
    classes: list[str]
    holdout_size: float | None
    train_on_test: bool

    def __init__(
        self,
        task_name: str,
        projector: NNProjector,
        preprocessor_name: PreprocessorEnum,
        train_batch_size: int = 32,
        test_batch_size: int = 32,
        train_on_test: bool = False,
        holdout_size: float | None = None,
    ):
        super().__init__()
        self.task_name = task_name
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.projector = projector
        self.preprocessor = PREPROCESSORS[preprocessor_name]
        self.dataset_name = "fact-40/pcbm_metashift"
        self.cache_dir = Path(".cache")
        self.logger = logging.getLogger()
        self.train_on_test = train_on_test
        self.holdout_size = holdout_size
        if self.train_on_test:
            assert holdout_size is not None, "Holdout split must be specified"
            self.logger.warning(
                f"Training on test set with holdout split of {holdout_size}"
            )

    def _load_dataset(self) -> datasets.DatasetDict:
        return (
            datasets.load_dataset(
                self.dataset_name, name=self.task_name, trust_remote_code=True
            )
            .map(self.convert_to_features, batched=True, remove_columns="image")
            .with_format("torch")
        )

    def _generate_holdout_size(self):
        """Generate holdout split for training on test set.
        Take 1 - `holdout_size` for each class of the test set for training
        and `holdout_size` for each class of the test set for validation and testing.

        Example:
            For an original test set with 100 images per class and a holdout_size of 0.2,
            the new training set will be 80 images per class
            and the new test set will be 20 images per class.
        """
        return self.dataset["test"].train_test_split(
            test_size=self.holdout_size,
            shuffle=True,
            seed=42,  # Should we let pytorch lightning handle the randomness?
            stratify_by_column="label",
        )

    def setup(self, stage: str):
        dataset_subset = f"{self.dataset_name}/{self.task_name}"
        cached_dataset_name = self.cache_dir / dataset_subset / self.projector.name
        try:
            self.logger.info(
                f"Attempting to load {dataset_subset} from cache {cached_dataset_name}"
            )
            self.dataset = datasets.load_from_disk(cached_dataset_name)
            self.logger.info(
                f"Loaded {dataset_subset} from cache {cached_dataset_name}"
            )
        except FileNotFoundError:
            self.logger.info(
                f"Failed to load {dataset_subset} from cache {cached_dataset_name}"
            )
            self.dataset = self._load_dataset()
            self.logger.info(f"Caching {dataset_subset} to {cached_dataset_name}")
            self.dataset.save_to_disk(cached_dataset_name)
            self.logger.info(
                f"Cached {self.dataset_name}/{self.task_name} to {cached_dataset_name}"
            )
        self.classes = self.dataset["train"].info.features["label"].names
        if self.train_on_test:
            self.dataset = self._generate_holdout_size()

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


class MetaShiftDataModuleSK(MetaShiftDataModule):
    TRAIN_SIZE: int = 500
    TEST_SIZE: int = 500

    def __init__(
        self,
        task_name: str,
        projector: NNProjector,
        preprocessor_name: PreprocessorEnum,
        train_on_test: bool = False,
        holdout_size: float | None = None,
    ):
        super().__init__(
            task_name,
            projector,
            preprocessor_name,
            self.TRAIN_SIZE,
            self.TEST_SIZE,
            train_on_test,
            holdout_size,
        )
