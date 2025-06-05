from torch.utils.data import Dataset

class FlickrImageCaptionDataset(Dataset):
    """Create a dataset for Flickr30k."""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.index_map = []
        for img_idx, item in enumerate(dataset):
            for cap_idx in range(len(item['caption'])):
                self.index_map.append((img_idx, cap_idx))

    def __len__(self):
        return len(self.index_map)
        # return len(self.dataset)

    def __getitem__(self, idx):
        img_idx, caption_idx = self.index_map[idx]
        item = self.dataset[img_idx]
        image = item['image']
        caption = item['caption'][caption_idx]
        if self.transform:
            image = self.transform(image)
        return image, caption