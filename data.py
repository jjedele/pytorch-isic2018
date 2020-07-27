from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from pl_bolts.datamodules import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets.utils import download_and_extract_archive


class _Ham10kDataset(Dataset):
    def __init__(self, df):
        self._df = df

        self._transforms = transforms.Compose(
            [
                # transforms.Resize(265),
                # transforms.CenterCrop(224),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx):
        rec = self._df.iloc[idx]

        with open(rec.path, "rb") as image_file:
            img = Image.open(image_file)
            img = img.convert("RGB")

        return self._transforms(img), rec["class"]


_TRAIN_DATA_URL = "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_Input.zip"
_LABEL_DATA_URL = "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_GroundTruth.zip"


class Ham10kDataModule(LightningDataModule):

    def __init__(self, data_root="./dataset", batch_size=32):
        super().__init__()
        self._prepared = False
        self.data_root = Path(data_root)
        self.batch_size = batch_size

    def prepare_data(self, validation_split_fraction=0.1, test_split_fraction=0.1, random_seed=42):
        # TODO make these checks cover more cases
        if not self.data_root.is_dir():
            download_and_extract_archive(
                url=_TRAIN_DATA_URL,
                download_root=str(self.data_root),
                extract_root=str(self.data_root),
            )

            download_and_extract_archive(
                url=_LABEL_DATA_URL,
                download_root=str(self.data_root),
                extract_root=str(self.data_root),
            )

        image_root = self.data_root / "ISIC2018_Task3_Training_Input"

        self._image_root = image_root
        self._label_file = self.data_root / "ISIC2018_Task3_Training_GroundTruth" / "ISIC2018_Task3_Training_GroundTruth.csv"

        # prepare ground truth
        df = pd.read_csv(str(self._label_file), index_col=0)
        classes = sorted(list(df.columns))
        self.classes = classes

        # get single-column class
        def class_mapper(rec):
            clazz = next(t[0] for t in rec.items() if t[1] > 0)
            class_idx = classes.index(clazz)
            return class_idx

        df = df.apply(class_mapper, axis=1).to_frame(name="class")

        # prepare splits
        np.random.seed(random_seed)
        dev_fraction = validation_split_fraction + test_split_fraction
        train_idx, dev_idx = train_test_split(df.index, test_size=dev_fraction, stratify=df["class"], shuffle=True)
        val_idx, test_idx = train_test_split(dev_idx, test_size=validation_split_fraction/dev_fraction, stratify=df.loc[dev_idx]["class"], shuffle=True)

        # prepare dataframe
        df.loc[train_idx, "split"] = "training"
        df.loc[val_idx, "split"] = "validation"
        df.loc[test_idx, "split"] = "test"

        df["path"] = df.index.map(lambda img_id: str(image_root / f"{img_id}.jpg"))

        self._df = df

        print("Prepared HAM10000 Dataset:")
        print(f"Classes: {classes}")
        print(f"Training Split Size: {len(train_idx)}")
        print(f"Validation Split Size: {len(val_idx)}")
        print(f"Test Split Size: {len(test_idx)}")

        self._prepared = True

    def train_dataloader(self):
        assert self._prepared
        dataset = _Ham10kDataset(self._df.query("split=='training'"))
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        assert self._prepared
        dataset = _Ham10kDataset(self._df.query("split=='validation'"))
        return DataLoader(dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        assert self._prepared
        dataset = _Ham10kDataset(self._df.query("split=='test'"))
        return DataLoader(dataset, batch_size=self.batch_size)

    def class_weights(self, split="training"):
        assert self._prepared
        counts = self._df.query(f"split=='{split}'")["class"].value_counts()

        weights = pd.Series([0 for _ in range(len(counts))])

        for leason_type in counts.keys():
            others = [t for t in counts.keys() if t != leason_type]
            other_counts = counts[others]
            total = other_counts.product()
            weights[leason_type] = total / counts.sum()

        weights /= weights.sum()
        return torch.Tensor(weights.values)



if __name__ == "__main__":
    #ds = Ham10kDataset(
    #    "/Volumes/Extern Jeff/datasets/ham10000/HAM10000_metadata.csv",
    #    ["/Volumes/Extern Jeff/datasets/ham10000/images"],
    #    split="train",
    #)

    #print(len(ds))
    #print(ds[0])
    h10km = Ham10kDataModule()
    h10km.prepare_data()
    print(h10km.class_weights())
    train = h10km.train_dataloader()

    for i in train:
        print(i)
