import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils import normalize_input, compute_data_mean

class AnimeDataSet(Dataset):
    def __init__(self, args, transform=None):
        """   
        folder structure:
            - {data_dir}
                - photo
                    1.jpg, ..., n.jpg
                - {dataset}  # E.g Hayao
                    smooth
                        1.jpg, ..., n.jpg
                    style
                        1.jpg, ..., n.jpg
        """
        data_dir = args.data_dir
        dataset = args.dataset

        anime_dir = os.path.join(data_dir, dataset)
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f'Folder {data_dir} does not exist')

        if not os.path.exists(anime_dir):
            raise FileNotFoundError(f'Folder {anime_dir} does not exist')

        self.mean = compute_data_mean(os.path.join(anime_dir, 'style'))
        print(f'Mean(B, G, R) of {dataset} are {self.mean}')

        self.debug_samples = args.debug_samples or 0
        self.data_dir = data_dir
        self.image_files =  {}
        self.photo = 'train_photo'
        self.style = f'{anime_dir}/style'
        self.smooth =  f'{anime_dir}/smooth'
        self.gray_zeros = torch.zeros(3, 256, 256)

        for opt in [self.photo, self.style, self.smooth]:
            folder = os.path.join(data_dir, opt)
            files = os.listdir(folder)

            self.image_files[opt] = [os.path.join(folder, fi) for fi in files]

        self.transform = transform

        print(f'Dataset: real {len(self.image_files[self.photo])} style {self.len_anime}, smooth {self.len_smooth}')

    def __len__(self):
        return self.debug_samples or len(self.image_files[self.photo])

    @property
    def len_anime(self):
        return len(self.image_files[self.style])

    @property
    def len_smooth(self):
        return len(self.image_files[self.smooth])

    def __getitem__(self, index):
        image, _ = self.load_images(index, self.photo)
        anm_idx = index
        if anm_idx > self.len_anime - 1:
            anm_idx -= self.len_anime * (index // self.len_anime)

        anime, anime_gray = self.load_images(anm_idx, self.style)
        _, smooth_gray = self.load_images(anm_idx, self.smooth)

        return image, anime, anime_gray, smooth_gray

    def load_images(self, index, opt):
        is_style = opt in {self.style, self.smooth}
        fpath = self.image_files[opt][index]
        image = cv2.imread(fpath)[:,:,::-1]

        if is_style:
            image_gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
            image_gray = np.stack([image_gray, image_gray, image_gray], axis=-1)
            image_gray = self._transform(image_gray, addmean=False)
            image_gray = image_gray.transpose(2, 0, 1)
            image_gray = torch.tensor(image_gray)
        else:
            image_gray = self.gray_zeros

        image = self._transform(image, addmean=is_style)
        image = image.transpose(2, 0, 1)

        return torch.tensor(image), image_gray


    def _transform(self, img, addmean=True):
        if self.transform is not None:
            img =  self.transform(image=img)['image']

        img = img.astype(np.float32)
        if addmean:
            img += self.mean
    
        return normalize_input(img)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    anime_loader = DataLoader(AnimeDataSet('dataset/Hayao/smooth'), batch_size=2, shuffle=True)

    img, img_gray = iter(anime_loader).next()
    plt.imshow(img[1].numpy().transpose(1, 2, 0))
    plt.show()
