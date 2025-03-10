from pathlib import Path
from loguru import logger
from typing import Union, Annotated, Literal, Tuple, List
import shutil
import typer
from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from mlops_project_tcs.crop_img import CropExtremePoints
from mlops_project_tcs.utils import plot_image_grid_with_labels, get_targets_from_subset
from hashlib import md5
import random
import sys

app = typer.Typer()


class BinaryClassBalancer:
    def __init__(self, datadir: Union[Path, str], output_dir: Union[Path, str]) -> None:
        """
        Initialize the BinaryClassBalancer class.

        Args:
            datadir (Union[Path, str]): Path to the dataset directory containing 'yes' and 'no' subdirectories.
            output_dir (Union[Path, str]): Path where balanced dataset will be saved.
        """
        self.datadir = Path(datadir)
        self.output_dir = Path(output_dir)
        self.classes = ["yes", "no"]
        self.file_paths = {cls: [] for cls in self.classes}
        self.file_hashes = set()

    def _compute_file_hash(self, file_path: Path) -> str:
        """
        Compute a hash for the file content to identify duplicates.

        Args:
            file_path (Path): Path to the file.

        Returns:
            str: Hash of the file content.
        """
        with open(file_path, "rb") as f:
            return md5(f.read()).hexdigest()

    def load_files(self) -> None:
        """
        Load files from the 'yes' and 'no' subdirectories, filtering out duplicates.
        """
        for cls in self.classes:
            class_dir = self.datadir / cls
            if not class_dir.exists():
                raise FileNotFoundError(f"Class directory '{class_dir}' does not exist.")

            for file_path in class_dir.iterdir():
                if not file_path.is_file():
                    continue

                file_hash = self._compute_file_hash(file_path)

                if file_hash not in self.file_hashes:
                    self.file_paths[cls].append(file_path)
                    self.file_hashes.add(file_hash)
                else:
                    logger.info(f"File: {file_path} is a duplicate. Skipping this file")

    def identify_minority_class(self) -> Tuple[str, str]:
        """
        Identify the minority class based on file counts.

        Returns:
            Tuple[str, str]: Minority and majority class names.
        """
        counts = {cls: len(self.file_paths[cls]) for cls in self.classes}
        minority_class = min(counts, key=counts.get)
        majority_class = max(counts, key=counts.get)

        # This if statement avoids a bug when equally sized classes, resulting in minority and majority being the same class
        if minority_class == majority_class:
            majority_class = [cls for cls in self.classes if cls != majority_class][0]

        return minority_class, majority_class

    def balance_and_write(self) -> None:
        """
        Balance the dataset and write to the output directory.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for cls in self.classes:
            (self.output_dir / cls).mkdir(parents=True, exist_ok=True)

        minority_class, majority_class = self.identify_minority_class()
        minority_files = self.file_paths[minority_class]
        majority_files = self.file_paths[majority_class]

        balanced_majority_files = majority_files[: len(minority_files)]

        # Combine the files to balance the dataset
        balanced_files = {minority_class: minority_files, majority_class: balanced_majority_files}

        # Copy the files to the output directory
        for cls, files in balanced_files.items():
            for file_path in files:
                dest_path = self.output_dir / cls / file_path.name
                shutil.copy2(file_path, dest_path)

        logger.info("Dataset balanced!")
        logger.info(f"Number of files in '{minority_class}': {len(minority_files)}")
        logger.info(f"Number of files in '{majority_class}': {len(balanced_majority_files)}")
        logger.info(f"Files saved to {self.output_dir}")

    def execute(self) -> None:
        """
        Main method to load, balance, and write files.
        """
        logger.info("Loading files...")
        self.load_files()

        logger.info("Balancing dataset...")
        self.balance_and_write()


class ImageAugmenter:
    def __init__(self, datadir: Union[Path, str], output_dir: Union[Path, str]):
        """
        Initialize the ImageAugmenter class.

        Args:
            datadir (Union[Path, str]): Path to the dataset directory containing 'yes' and 'no' subdirectories.
            output_dir (Union[Path, str]): Path where augmented images will be saved.
        """
        self.datadir = Path(datadir)
        self.output_dir = Path(output_dir)
        self.classes = ["no", "yes"]
        self.image_paths = {cls: self._load_image_paths(cls) for cls in self.classes}
        self.class_counts = {cls: len(self.image_paths[cls]) for cls in self.classes}
        self.total_count = sum(self.class_counts.values())

        # Ensure output directories exist
        for cls in self.classes:
            (self.output_dir / cls).mkdir(parents=True, exist_ok=True)

    def _load_image_paths(self, cls: str):
        """
        Load image paths for a given class.

        Args:
            cls (str): Class name ('yes' or 'no').

        Returns:
            List[Path]: List of paths to images.
        """
        class_dir = self.datadir / cls
        return [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}]

    def _augment_image(self, image_path: Path):
        """
        Augment an image by applying random rotation.

        Args:
            image_path (Path): Path to the image to augment.

        Returns:
            Image.Image: Augmented PIL image.
        """
        image = Image.open(image_path).convert("RGB")
        angle = random.uniform(-30, 30)  # Random rotation angle between -30 and 30 degrees
        return image.rotate(angle, resample=Image.BICUBIC, expand=True)

    def _save_image(self, image: Image.Image, cls: str):
        """
        Save the augmented image to the output directory.

        Args:
            image (Image.Image): Augmented PIL image.
            cls (str): Class name ('yes' or 'no').
        """
        filename = f"aug_{random.randint(100000, 999999)}.jpg"
        while (self.output_dir / cls / filename).exists():
            filename = f"aug_{random.randint(100000, 999999)}.jpg"
        save_path = self.output_dir / cls / filename
        image.save(save_path)

    def augment(self, iterations: int = 10):
        """
        Augment the dataset by adding new images to balance classes.

        Args:
            iterations (int): Number of augmentation iterations to perform augmentation for each class.
        """
        for cls in self.classes:
            for _ in range(iterations):
                image_path = random.choice(self.image_paths[cls])
                augmented_image = self._augment_image(image_path)
                self._save_image(augmented_image, cls)

                # Update class counts
                self.class_counts[cls] += 1


class BrainMRIDataset(Dataset):
    """Custom dataset for loading and preprocessing brain MRI images."""

    def __init__(self, datadir: Union[Path, str]) -> None:
        """
        Initialize the BrainMRIDataset class.

        Args:
            datadir (Union[Path, str]): Path to the dataset directory.
        """
        self.datadir = Path(datadir)
        self.image_paths: List[Path] = []
        self.labels: List[int] = []
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ],
        )
        self.load_image_paths(self.datadir)

    def load_image_paths(self, datadir: Union[Path, str]) -> None:
        """
        Load image paths and labels from the dataset directory.

        Args:
            datadir (Union[Path, str]): Path to the dataset directory.
        """
        datadir = Path(datadir)
        for label, folder in enumerate(["no", "yes"]):
            category_path = datadir / folder
            for fname in category_path.iterdir():
                if fname.is_file() and fname.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self.image_paths.append(fname)
                    self.labels.append(label)

    def __len__(self) -> int:
        """
        Return the length of the dataset.

        Returns:
            int: Number of images in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Return a given sample from the dataset.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, int]: Tuple containing the image tensor and its label.
        """
        img_path = self.image_paths[index]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = self.labels[index]
        return image, label

    def preprocess(self, output_folder: Union[Path, str]) -> None:
        """
        Preprocess the raw data by cropping and resizing. Saves preprocessed data to given directory by creating 'yes' and 'no' subdirectories.

        Args:
            output_folder (Union[Path, str]): Path to the output directory.
        """
        cropper = CropExtremePoints(add_pixels_value=10, target_size=(224, 224))
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        for img_path, label in zip(self.image_paths, self.labels):
            image = Image.open(img_path).convert("RGB")
            processed_image = cropper(image)
            label_folder = output_folder / ("no" if label == 0 else "yes")
            label_folder.mkdir(parents=True, exist_ok=True)

            output_dir = label_folder / img_path.name
            processed_image_pil = transforms.ToPILImage()(processed_image)
            processed_image_pil.save(output_dir)
            logger.debug(f"Processed and saved: {output_dir}")


class BrainMRIDataModule(pl.LightningDataModule):
    """Pytorch Lightning Data module for loading and preprocessing brain MRI images."""

    def __init__(
        self,
        datadir: Union[Path, str],
        mode: Literal["train", "data_prep"] = "train",
        batch_size: int = 10,
        val_split: float = 0.2,
        test_split: float = 0.1,
        num_workers: int = 4,
    ) -> None:
        """
        Initialize the BrainMRIDataModule class.

        Args:
            datadir (Union[Path, str]): Path to the dataset directory. In 'train' mode this directory should point to the parent directory of directories with 'train', 'val', 'test'. For data_prep mode, this should point to the directory with 'yes' and 'no' subdirectories containing raw img files.
            mode (Literal["train", "data_prep"]): Mode of operation, either 'train' or 'data_prep'.
            batch_size (int): Batch size for data loading.
            val_split (float): Fraction of data to use for validation.
            test_split (float): Fraction of data to use for testing.
            num_workers (int): Number of worker processes for data loading.
        """
        super().__init__()
        self.datadir = Path(datadir)
        self.mode = mode
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.num_workers = num_workers
        self.prep_dataset = None

        if self.mode == "data_prep":
            self.data_prep_split_dataset()

    def data_prep_split_dataset(self) -> None:
        """
        Split the dataset into training, validation, and test sets.
        """
        self.prep_dataset = BrainMRIDataset(self.datadir)

        test_size = int(len(self.prep_dataset) * self.test_split)
        val_size = int(len(self.prep_dataset) * self.val_split)
        train_size = len(self.prep_dataset) - test_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.prep_dataset, [train_size, val_size, test_size]
        )

    def setup(self, stage: str = None) -> None:
        """
        Set up the datasets for training, validation, and testing.

        Args:
            stage (str, optional): Stage of the setup process.
        """
        if self.mode != "train":
            logger.error(
                f"Datamodule mode is set to {self.mode} during training attempt. The data module does not allow this mode during training run."
            )
            sys.exit(1)

        self.train_dataset = BrainMRIDataset(self.datadir / "train")
        self.val_dataset = BrainMRIDataset(self.datadir / "val")
        self.test_dataset = BrainMRIDataset(self.datadir / "test")

    def train_dataloader(self) -> DataLoader:
        """
        Return the training data loader.

        Returns:
            DataLoader: Training data loader.
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        """
        Return the validation data loader.

        Returns:
            DataLoader: Validation data loader.
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        """
        Return the test data loader.

        Returns:
            DataLoader: Test data loader.
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def save_splits(self, output_dir: Union[Path, str]):
        """
        Save the train, val, and test splits into subdirectories in a new folder.

        Args:
            output_dir (Union[Path, str]): Path to the output directory.
        """
        if self.mode != "data_prep":
            logger.warning(
                f"Trying to run save_splits() function with {self}, when in mode {self.mode}. This is not possible."
            )
            return

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        splits = {"train": self.train_dataset, "val": self.val_dataset, "test": self.test_dataset}

        """
        for split_name, dataset in splits.items():
            split_dir = output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            for i, (image, label) in enumerate(dataset):
                string_label = "no" if label == 0 else "yes"
                label_folder = split_dir / string_label
                label_folder.mkdir(parents=True, exist_ok=True)
                output_path = label_folder / f"{string_label}_{split_name}_{i}.jpg"
                image_pil = transforms.ToPILImage()(image)
                image_pil.save(output_path)
                logger.info(f"Saved {split_name} image: {output_path}")
        """
        for split_name, datasubset in splits.items():
            split_dir = output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            for index in datasubset.indices:
                label = self.prep_dataset.labels[index]
                image_path = self.prep_dataset.image_paths[index]
                string_label = "no" if label == 0 else "yes"
                label_folder = split_dir / string_label
                label_folder.mkdir(parents=True, exist_ok=True)
                output_path = label_folder / f"{string_label}_{split_name}_{index}.jpg"
                shutil.copy(image_path, output_path)
                logger.info(f"Saved {split_name} image: {image_path} --> {output_path}")


@app.command()
def balance(
    datadir: Annotated[
        str, typer.Option("--indir", "-i", help="Input directory of data containing 'yes' and 'no' subdirectories")
    ] = "data/raw",
    output_folder: Annotated[
        str, typer.Option("--outdir", "-o", help="Output directory of balanced data files")
    ] = "data/balanced",
    seed: Annotated[int, typer.Option("--seed", "-s", help="Seed for reproducibility")] = 123,
) -> None:
    """Data balancing by sub-sampling to fit minority class."""
    logger.add("logs/data_balance_data.log", level="DEBUG")
    logger.info("Running data balancer...")
    class_balancer = BinaryClassBalancer(datadir=Path(datadir), output_dir=Path(output_folder))
    class_balancer.execute()


@app.command()
def split(
    datadir: Annotated[
        str, typer.Option("--indir", "-i", help="Input directory of data containing 'yes' and 'no' subdirectories")
    ] = "data/balanced",
    output_folder: Annotated[
        str, typer.Option("--outdir", "-o", help="Output directory of data splits files")
    ] = "data/split",
    seed: Annotated[
        int, typer.Option("--seed", "-s", help="Seed for reproducibility via pl.seed_everything(seed)")
    ] = 123,
) -> None:
    """Splitting data image files into train, validation and test splits."""
    logger.add("logs/data_split_data.log", level="DEBUG")
    logger.info("Preprocessing data...")
    pl.seed_everything(seed)
    data_module = BrainMRIDataModule(Path(datadir), mode="data_prep")
    data_module.save_splits(Path(output_folder))


@app.command()
def augment(
    datadir: Annotated[
        str,
        typer.Option(
            "--indir",
            "-i",
            help="Input directory of data containing 'train', 'validation' and 'test' directories, with 'yes' and 'no' subdirectories",
        ),
    ] = "data/split",
    output_folder: Annotated[
        str, typer.Option("--outdir", "-o", help="Output directory of data splits files")
    ] = "data/split_augmented",
    scale_ratio: Annotated[
        float,
        typer.Option(
            "--scale_ratio",
            "-sr",
            help="The ratio of scaling for augmentation (value of 2 will double the dataset size). Requires scale_ratio > 1",
        ),
    ] = 2,
    seed: Annotated[int, typer.Option("--seed", "-s", help="Seed for reproducibility via random.seed()")] = 123,
) -> None:
    """Augments split datasets by random rotation"""
    logger.add("logs/data_augment.log", level="DEBUG")
    logger.info("Augmenting data...")
    random.seed(seed)
    datadir = Path(datadir)
    output_folder = Path(output_folder)

    if scale_ratio < 1:
        logger.warning("scale_ratio must be above 1 in order for augmentation to run. Skipping augmentation")
        sys.exit(1)

    scale_ratio = scale_ratio - 1  # Correct scaling since we copy the original files

    for split in ["train", "val", "test"]:
        shutil.copytree(datadir / split, output_folder / split, dirs_exist_ok=True)
        dataset = BrainMRIDataset(datadir=datadir / split)
        augmenter = ImageAugmenter(datadir=datadir / split, output_dir=output_folder / split)
        augmenter.augment(iterations=int(len(dataset) * scale_ratio))
        logger.info(f"Augmented {split}-split from {len(dataset)} to {len(dataset) * (scale_ratio + 1)}")


@app.command()
def preprocess(
    datadir: Annotated[
        str,
        typer.Option(
            "--indir",
            "-i",
            help="Input directory of data containing 'train', 'validation' and 'test' directories, with 'yes' and 'no' subdirectories",
        ),
    ] = "data/split_augmented",
    output_folder: Annotated[
        str, typer.Option("--outdir", "-o", help="Output directory of preprocessed files")
    ] = "data/processed",
    seed: Annotated[
        int, typer.Option("--seed", "-s", help="Seed for reproducibility via pl.seed_everything(seed)")
    ] = 123,
) -> None:
    """Preprocessing image data by cropping and resizing."""
    logger.add("logs/data_preprocess.log", level="DEBUG")
    logger.info("Preprocessing data...")
    pl.seed_everything(seed)
    datadir = Path(datadir)
    output_folder = Path(output_folder)
    for split in ["train", "val", "test"]:
        dataset = BrainMRIDataset(datadir=datadir / split)
        dataset.preprocess(output_folder=output_folder / split)


@app.command()
def dataset_statistics(
    datadir: Annotated[
        str,
        typer.Option(
            "--datadir", "-i", help="Directory containing kaggle brain imaging data with 'yes' and 'no' subdirectories"
        ),
    ] = "data/processed",
    savedir: Annotated[
        str, typer.Option("--savedir", "-o", help="Directory for saving statistic figures")
    ] = "reports/data_statistics",
    seed: Annotated[
        int, typer.Option("--seed", "-s", help="Seed for reproducibility via pl.seed_everything(seed)")
    ] = 123,
) -> None:
    """Printing dataset statistics."""
    savedir = Path(savedir)
    savedir.mkdir(parents=True, exist_ok=True)
    pl.seed_everything(seed)
    data_module = BrainMRIDataModule(datadir=datadir)
    data_module.setup()
    print("Train dataset")
    print(f"Number of images: {len(data_module.train_dataset)}")
    print(f"Image shape: {data_module.train_dataset[0][0].shape}")
    print("\n")
    print("Validation dataset")
    print(f"Number of images: {len(data_module.val_dataset)}")
    print(f"Image shape: {data_module.val_dataset[0][0].shape}")
    print("\n")
    print("Test dataset")
    print(f"Number of images: {len(data_module.test_dataset)}")
    print(f"Image shape: {data_module.test_dataset[0][0].shape}")

    plot_images = []
    plot_labels = []
    for _ in range(25):
        index = random.randint(0, len(data_module.test_dataset.image_paths) - 1)
        image = data_module.test_dataset.image_paths[index]
        label = data_module.test_dataset.labels[index]

        plot_images.append(image)
        plot_labels.append(label)
    plot_image_grid_with_labels(image_paths=plot_images, rows=5, cols=5, labels=plot_labels)
    plt.savefig(savedir / "kaggle_example_images.png")
    plt.close()

    train_label_distribution = torch.bincount(get_targets_from_subset(data_module.train_dataset))
    val_label_distribution = torch.bincount(get_targets_from_subset(data_module.val_dataset))
    test_label_distribution = torch.bincount(get_targets_from_subset(data_module.test_dataset))

    plt.bar(torch.arange(2), train_label_distribution)
    plt.title("Train label distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig(savedir / "train_label_distribution.png")
    plt.close()

    plt.bar(torch.arange(2), val_label_distribution)
    plt.title("Validation label distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig(savedir / "val_label_distribution.png")
    plt.close()

    plt.bar(torch.arange(2), test_label_distribution)
    plt.title("Test label distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig(savedir / "test_label_distribution.png")
    plt.close()


if __name__ == "__main__":
    app()
