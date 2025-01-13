from loguru import logger
from typing import Union, Annotated
from pathlib import Path
import os
import shutil
from PIL import Image
from torch.utils.data import Dataset
import kagglehub
import typer
from torchvision import transforms  # <-- Add this import
from crop_img import CropExtremePoints


app = typer.Typer()
logger.add("logs/data_log.log", level="DEBUG")

def get_kaggle_dataset(
        kaggle_handle: str,
        raw_data_dir: Union[Path, str]
) -> None:
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

class MyDataset(Dataset):
    """Custom dataset for preprocessing brain MRI images."""

    def __init__(self, raw_data_path: Path) -> None:
        self.data_path = raw_data_path
        self.image_paths = []
        self.labels = []

        for label, folder in enumerate(['no', 'yes']):
            folder_path = os.path.join(raw_data_path, folder)
            for fname in os.listdir(folder_path):
                if fname.endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(folder_path, fname))
                    self.labels.append(label)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.image_paths)

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        img_path = self.image_paths[index]
        label = self.labels[index]

        image = Image.open(img_path).convert("RGB")
        return image, label

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
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

@app.command()
def download(
    kaggle_handle: Annotated[str, typer.Option("--kaggle_handle", "-kh", help="Kaggle handle pointing to download location")] = "navoneel/brain-mri-images-for-brain-tumor-detection",
    raw_data_dir: Annotated[str, typer.Option("--output_dir", "-o", help="Output directory of the downloaded Kaggle data")] = "data/raw",
) -> None:
    logger.info("Downloading and preparing raw data...")
    get_kaggle_dataset(kaggle_handle, raw_data_dir)


@app.command()
def preprocess(
    raw_data_dir: Annotated[str, typer.Option("--indir", "-i", help="Input directory of raw data")] = "data/raw",
    output_folder: Annotated[str, typer.Option("--outdir", "-o", help="Output directory of preprocessed files")] = "data/processed"
) -> None:
    logger.info("Preprocessing data...")
    dataset = MyDataset(Path(raw_data_dir))
    dataset.preprocess(Path(output_folder))


if __name__ == "__main__":
    app()