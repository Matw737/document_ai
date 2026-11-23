from torch.utils.data import Dataset
from torchvision import transforms
import torch
from PIL import Image
import torch.nn.functional as F


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

        ratio = 26/ height
        new_width = round((width*ratio))

        img = img.resize((new_width,26))

        img = transforms.ToTensor()(img)

        return img, index

def collate_fn_inference(batch):
   
    BORDER_PAD = 3
    
    padded_batch = []
    lengths = []
    
    for img, index in batch:
       
        img_padded = F.pad(img, (BORDER_PAD, BORDER_PAD, BORDER_PAD, BORDER_PAD), mode='constant', value=1.0)
        
        padded_batch.append((img_padded, index))
        lengths.append(img_padded.shape[-1]) 

    max_width = max(lengths)
    images = []
    indices = []

    for img, index in padded_batch:
        delta = max_width - img.shape[-1]
        
        if delta > 0:
        
            right_padding = torch.ones(1, img.shape[1], delta, dtype=img.dtype)
            img = torch.cat([img, right_padding], dim=2)

        images.append(img)
        indices.append(index)

    images = torch.stack(images)
    input_lengths = torch.tensor(lengths, dtype=torch.long)
    indices = torch.tensor(indices, dtype=torch.long)

    return images, input_lengths, indices
