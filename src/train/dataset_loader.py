from torch.utils.data import Dataset
from PIL import Image
from enum import Enum
import pathlib
from tqdm import tqdm


class DatasetMode(str, Enum):
    TRAIN = "train"
    VAL = "valid"
    TEST = "test"


class ObjectDetectionDataset(Dataset):
    def __init__(self, data_dir: str, mode: DatasetMode, transform=None):
        self.data_dir = pathlib.Path(data_dir)
        self.mode = mode
        self.transform = transform
        self.classes = self.load_classes()
        self.image_files = [
            str(f) for f in (self.data_dir / "images" / self.mode).glob("*.jpg")
        ]
        self.labels = self.load_labels()

    def load_classes(self):
        with open(self.data_dir / "classes.txt", "r") as f:
            classes = f.readlines()
        classes = [c.strip() for c in classes]
        return classes

    def load_labels(self):
        labels = {}
        for image_file in tqdm(
            self.image_files, desc="Loading labels", total=len(self.image_files)
        ):
            image_file = pathlib.Path(image_file)
            label_file = self.data_dir / "labels" / self.mode / f"{image_file.stem}.txt"
            with open(label_file, "r") as f:
                label = f.readlines()
            label = [l.strip().split() for l in label]
            class_id, bbox = int(label[0][0]), [float(x) for x in label[0][1:]]
            labels[label_file.stem] = (class_id, bbox)
        return labels

    def __len__(self):
        # get number of images in the dataset folder
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_file = pathlib.Path(image_file)
        image = Image.open(image_file)
        class_id, bbox = self.labels[image_file.stem]
        if self.transform:
            image = self.transform(image)
        return image, class_id, bbox


if __name__ == "__main__":
    # Example usage
    data_dir = "D:\\Projects\\ml-ops-wildlife\\data\\WAID"
    dataset = ObjectDetectionDataset(data_dir, DatasetMode.TRAIN)
    print(len(dataset))
    print(dataset[0])

    # from torch.utils.data import DataLoader

    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
