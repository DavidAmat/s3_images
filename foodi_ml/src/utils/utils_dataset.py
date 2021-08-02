import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class ImageEmbeddingsDatasetPrediction(Dataset):
    def __init__(self, samples: pd.DataFrame, tag=None):
        """
        samples: pd.DataFrame -> ['store_name', 'product_name', 'collection_name',
        'product_descr', 'image_url', 'image_service_id', 'all_text']
        """
        self.samples = samples
        self.img_size = (224, 224)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),  #  3 x H' x W'
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.tag = tag

    def get_samples(self):
        return self.samples

    def __getitem__(self, idx):
        img_ori = self.transform(
            Image.open(self.samples["img_path"].iloc[idx]).convert("RGB")
        )
        image_service_id = self.samples["image_service_id"].iloc[idx]
        product_id = self.samples["product_id"].iloc[idx]
        # image_names_s3 has the format artifacts/<version>/<city_code>/images/XXXXX_yyyyy.png
        city_code = (
            self.tag
            if self.tag is not None
            else self.samples["image_names_s3"].iloc[idx].split("/")[-3]
        )
        return img_ori, image_service_id, int(product_id.item()), city_code

    def __len__(self):
        return self.samples.shape[0]