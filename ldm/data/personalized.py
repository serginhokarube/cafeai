import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import random

per_img_token_list = [
    'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'כ', 'ל', 'מ', 'נ', 'ס', 'ע', 'פ', 'צ', 'ק', 'ר', 'ש', 'ת',
]

class PersonalizedBase(Dataset):
    def __init__(self,
                 data_root,
                 size=None,
                 repeats=100,
                 interpolation="bicubic",
                 flip_p=0.5,
                 set="train",
                 placeholder_token="*",
                 per_image_tokens=False,
                 center_crop=False,
                 mixing_prob=0.25,
                 coarse_class_text=None,
                 templates=[],
                 ):

        self.data_root = data_root

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]

        # self._length = len(self.image_paths)
        self.num_images = len(self.image_paths)
        self._length = self.num_images

        self.placeholder_token = placeholder_token

        self.per_image_tokens = per_image_tokens
        self.center_crop = center_crop
        self.mixing_prob = mixing_prob

        self.coarse_class_text = coarse_class_text
        self.templates = templates

        if per_image_tokens:
            assert self.num_images < len(per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."

        if set == "train":
            self._length = self.num_images * repeats

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i % self.num_images])

        placeholder_string = self.placeholder_token
        if self.coarse_class_text:
            placeholder_string = f"{self.coarse_class_text} {placeholder_string}"

        image = image.convert('RGBA')
        new_image = Image.new('RGBA', image.size, 'WHITE')
        new_image.paste(image, (0, 0), image)
        image = new_image.convert('RGB')

        #filename = os.path.basename(self.image_paths[i % self.num_images])
        #filename_tokens = os.path.splitext(filename)[0].replace(' ', '-').replace('_', '-').split('-')
        #filename_tokens = [token for token in filename_tokens if token.isalpha()]

        if self.per_image_tokens and np.random.uniform() < self.mixing_prob:
            text = random.choice(self.templates).format(placeholder_string, per_img_token_list[i % self.num_images])
        else:
            text = random.choice(self.templates).format(placeholder_string)

        example["caption"] = text

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = img.shape[0], img.shape[1]
            img = img[(h - crop) // 2:(h + crop) // 2,
                (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        return example