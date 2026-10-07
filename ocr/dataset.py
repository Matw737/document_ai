from torch.utils.data import Dataset
from torchvision import transforms
import torch
from PIL import Image

class OCRInferenceDataset(Dataset):

    def __init__(self, crop_records):
        self.crop_records = crop_records

    def __len__(self):
        return len(self.crop_records)

    def __getitem__(self, index):

        record = self.crop_records[index]

        image_path = record["path"]

        img = Image.open(image_path)

        width, height = img.size

        img = img.convert("L")

        ratio = 32 / height
        new_width = round((width*ratio))

        img = img.resize((new_width,32))

        img = transforms.ToTensor()(img)

        return img, index

def collate_fn_inference(batch):

    lengths = [
        img.shape[-1]
        for img,_ in batch
    ]

    max_width = max(lengths)

    images = []
    indices = []

    for img, index in batch:

        delta = max_width - img.shape[-1]

        padding = torch.zeros(1, 32, delta, dtype=img.dtype)

        img = torch.cat([img, padding], dim=2)

        images.append(img)
        indices.append(index)

    images = torch.stack(images)

    input_lengths = torch.tensor(
        lengths,
        dtype=torch.long
    )
    indices = torch.tensor(
        indices,
        dtype=torch.long
    )

    return images, input_lengths, indices