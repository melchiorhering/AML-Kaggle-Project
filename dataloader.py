#  Importing Necessary Libraries for a PyTorch project
import os
from torch.utils.data import Dataset
from PIL import Image
import polars as pl


# Custom dataset class
class BirdDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.annotations = pl.read_csv(
            csv_file
        ).to_pandas()  # Convert to Pandas DataFrame
        self.transform = transform

        # Check labels and paths during initialization
        if self.annotations["label"].min() < 1 or self.annotations["label"].max() > 200:
            raise ValueError("Found label outside the range [1, 200]")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = f".{self.annotations.iloc[index, 0]}"
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = Image.open(img_path).convert("RGB")
        label = int(self.annotations.iloc[index, 1]) - 1  # Adjust labels to [0, 199]

        if label < 0 or label >= 200:
            raise ValueError(f"Adjusted label out of range: {label}")

        if self.transform:
            image = self.transform(image)

        return image, label
