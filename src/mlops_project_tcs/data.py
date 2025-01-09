from pathlib import Path
import os
import shutil
from PIL import Image
from torch.utils.data import Dataset
import kagglehub
import typer
from torchvision import transforms  # <-- Add this import
from crop_img import CropExtremePoints

def download_raw_data(raw_data_path: Path) -> None:
    """Download dataset using kagglehub and place it in the specified raw data folder."""
    # Download the dataset
    downloaded_path = kagglehub.dataset_download("navoneel/brain-mri-images-for-brain-tumor-detection")
    print(f"Dataset downloaded to: {downloaded_path}")

    # Ensure the target folder exists
    raw_data_path.mkdir(parents=True, exist_ok=True)

    # Identify the folder that contains the "yes" and "no" subfolders
    dataset_folder = Path(downloaded_path)  # Assuming the downloaded dataset has a single folder
    for subfolder in os.listdir(dataset_folder):
        if subfolder.lower() == "brain_tumor_dataset":
            brain_tumor_dataset_path = dataset_folder / subfolder
            break

    # Check if the dataset folder was found
    if 'brain_tumor_dataset_path' in locals():
        print(f"Copying dataset contents from {brain_tumor_dataset_path} to {raw_data_path}...")
        shutil.copytree(brain_tumor_dataset_path, raw_data_path, dirs_exist_ok=True)
        print(f"Dataset successfully placed in: {raw_data_path}")
    else:
        raise ValueError("Expected 'brain_tumor_dataset' folder not found in the downloaded dataset.")


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
            print(f"Processed and saved: {output_path}")

def preprocess(raw_data_path: Path, output_folder: Path) -> None:
    print("Downloading and preparing raw data...")
    download_raw_data(raw_data_path)

    print("Preprocessing data...")
    dataset = MyDataset(raw_data_path)
    dataset.preprocess(output_folder)

if __name__ == "__main__":
    typer.run(preprocess)
