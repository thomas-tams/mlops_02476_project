from pathlib import Path
from loguru import logger
from typing import Union, Annotated
import shutil
import typer
import kagglehub
from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from mlops_project_tcs.crop_img import CropExtremePoints
from mlops_project_tcs.utils import plot_image_grid_with_labels, get_targets_from_subset

app = typer.Typer()
logger.add("logs/data_log.log", level="DEBUG")


def get_kaggle_dataset(kaggle_handle: str, raw_data_dir: Union[Path, str]) -> None:
    """Download dataset using kagglehub and place it in the specified raw data folder."""
    try:
        # Access data from kaggle
        downloaded_path = kagglehub.dataset_download(kaggle_handle)
        logger.info(f"Initial path to downloaded kaggle data: {downloaded_path}")
    except Exception as e:
        logger.error(f"Failed to download dataset from Kaggle: {e}")
        return

    # Removing double data (for some reason this kaggle dataset contains double up on the dataset)
    if kaggle_handle == "navoneel/brain-mri-images-for-brain-tumor-detection":
        downloaded_path = Path(downloaded_path)
        brain_tumor_dataset_path = downloaded_path / "brain_tumor_dataset"
        if brain_tumor_dataset_path.exists() and brain_tumor_dataset_path.is_dir():
            shutil.rmtree(brain_tumor_dataset_path)
            logger.info(f"Removed directory: {brain_tumor_dataset_path}")

    # Ensure the target folder exists
    raw_data_dir = Path(raw_data_dir)
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    # Copy files to raw directory
    logger.info(f"Copying dataset contents from {downloaded_path} to {raw_data_dir}...")
    shutil.copytree(downloaded_path, raw_data_dir, dirs_exist_ok=True)
    logger.info(f"Dataset successfully placed in: {raw_data_dir}")


class BrainMRIDataset(Dataset):
    """Custom dataset for loading and preprocessing brain MRI images."""

    def __init__(self, datadir: Path) -> None:
        self.datadir = datadir
        self.image_paths = []
        self.labels = []
        self.transform = transforms.ToTensor()
        self.load_image_paths(self.datadir)

    def load_image_paths(self, datadir: Union[Path, str]):
        datadir = Path(datadir)
        for label, folder in enumerate(["no", "yes"]):
            category_path = datadir / folder
            for fname in category_path.iterdir():
                if fname.is_file() and fname.suffix.lower() in {".jpg", ".jpeg", ".png", ".JPG"}:
                    self.image_paths.append(fname)
                    self.labels.append(label)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.image_paths)

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        img_path = self.image_paths[index]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = self.labels[index]
        return image, label

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data by cropping and resizing. Saves preprocessed data to given directory by creating 'yes' and 'no' subdirectories."""
        cropper = CropExtremePoints(add_pixels_value=10, target_size=(224, 224))
        output_folder.mkdir(parents=True, exist_ok=True)

        for img_path, label in zip(self.image_paths, self.labels):
            image = Image.open(img_path).convert("RGB")
            processed_image = cropper(image)  # Use cropper as callable
            label_folder = output_folder / ("no" if label == 0 else "yes")
            label_folder.mkdir(parents=True, exist_ok=True)

            output_path = label_folder / Path(img_path).name
            processed_image_pil = transforms.ToPILImage()(processed_image)  # Convert tensor to PIL Image
            processed_image_pil.save(output_path)
            logger.info(f"Processed and saved: {output_path}")


class BrainMRIDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datadir: Union[Path, str],
        batch_size: int = 10,
        val_split: float = 0.2,
        test_split: float = 0.1,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.datadir = Path(datadir)
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.num_workers = num_workers
        self.setup()

    def setup(self, stage: str = None):
        # Create the dataset
        dataset = BrainMRIDataset(self.datadir)

        # Split the dataset into training and validation sets
        test_size = int(len(dataset) * self.test_split)
        val_size = int(len(dataset) * self.val_split)
        train_size = len(dataset) - test_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        # Assuming the test dataset is set up similarly
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


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
    for i in range(25):
        image, label = data_module.test_dataset[i]
        plot_images.append(image)
        plot_labels.append(label)
    plot_image_grid_with_labels(images=plot_images, rows=5, cols=5, labels=plot_labels)
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


@app.command()
def download(
    kaggle_handle: Annotated[
        str, typer.Option("--kaggle_handle", "-kh", help="Kaggle handle pointing to download location")
    ] = "navoneel/brain-mri-images-for-brain-tumor-detection",
    raw_data_dir: Annotated[
        str, typer.Option("--output_dir", "-o", help="Output directory of the downloaded Kaggle data")
    ] = "data/raw",
) -> None:
    """Downloading kaggle data using kagglehub."""
    logger.info("Downloading and preparing raw data...")
    get_kaggle_dataset(kaggle_handle, raw_data_dir)


@app.command()
def preprocess(
    raw_data_dir: Annotated[
        str, typer.Option("--indir", "-i", help="Input directory of raw data containing 'yes' and 'no' subdirectories")
    ] = "data/raw",
    output_folder: Annotated[
        str, typer.Option("--outdir", "-o", help="Output directory of preprocessed files")
    ] = "data/processed",
) -> None:
    """Preprocessing kaggle image data by cropping and resizing."""
    logger.info("Preprocessing data...")
    dataset = BrainMRIDataset(Path(raw_data_dir))
    dataset.preprocess(Path(output_folder))


if __name__ == "__main__":
    app()
