import torch
import h5py
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms

class Shanghai(Dataset):
    def __init__(self, data_path, img_size, type="train", trans=None, seq_len=-1):
        super().__init__()

        self.data_path = data_path
        self.img_size = img_size

        assert type in ["train", "test", "val"]
        self.type = type if type != "val" else "test"
        with h5py.File(data_path, "r") as f:
            self.all_len = int(f[self.type]["all_len"][()])
        if trans is not None:
            self.transform = trans
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                ]
            )

    def __len__(self):
        return self.all_len

    def sample(self):
        index = np.random.randint(0, self.all_len)
        return self.__getitem__(index)

    def __getitem__(self, index):

        with h5py.File(self.data_path, "r") as f:
            imgs = f[self.type][str(index)][
                ()
            ]  # numpy array: (25, 565, 784), dtype=uint8, range(0,70)
            frames = torch.from_numpy(imgs).float().squeeze()
            frames = frames / 255.0
            frames = self.transform(frames)

        return {
            "sequence": frames[:5],
            "target": frames[5:10],
        }


# if __name__ == "__main__":
#     dataset = Shanghai("shanghai.h5", 128)
#     sample1 = dataset.sample()
#     print(sample1["sequence"].shape)
#
#     print(len(dataset))
