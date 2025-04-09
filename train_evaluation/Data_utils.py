import os
import pickle
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class DescriptionScene(Dataset):
    def __init__(self, data_description_path, mem, data_scene_path, customized_margin=False, verbose=True):
        self.description_path = data_description_path
        self.data_pov_path = data_scene_path
        available_data = open('data/available_data.txt', 'r')
        self.samples = [x[:-1] for x in available_data.readlines()]
        self.mem = mem
        self.margin_needed = customized_margin
        if self.mem:
            if verbose:
                print('Data Loading ...')

                print('Loading descriptions ...')
            if os.path.exists('./data/descs.pkl'):
                pickle_file = open('./data/descs.pkl', 'rb')
                self.descs = pickle.load(pickle_file)
                pickle_file.close()
            else:
                self.descs = []
                for idx, s in enumerate(self.samples):
                    self.descs.append(torch.load(self.description_path + os.sep + s + '.pt', weights_only=True))
                pickle_file = open('./data/descs.pkl', 'wb')
                pickle.dump(self.descs, pickle_file)
                pickle_file.close()
            if verbose:
                print('Loading POVs ...')
            if self.data_pov_path is not None:
                if os.path.exists('./data/pov_images.pkl'):
                    pickle_file = open('./data/pov_images.pkl', 'rb')
                    self.pov_images = pickle.load(pickle_file)
                    pickle_file.close()
                else:
                    self.pov_images = []
                    for idx, s in enumerate(self.samples):
                        self.pov_images.append(torch.load(self.data_pov_path + os.sep + s + '.pt', weights_only=True))
                    pickle_file = open('./data/pov_images.pkl', 'wb')
                    pickle.dump(self.pov_images, pickle_file)
                    pickle_file.close()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if self.mem:
            desc_tensor = self.descs[index]
            scene_img_tensor = self.pov_images[index]
        else:
            desc_tensor = torch.load(self.description_path + os.sep + self.samples[index] + '.pt')
            scene_img_tensor = torch.load(self.data_pov_path + os.sep + self.samples[index] + '.pt')
        if self.margin_needed:
            return desc_tensor, scene_img_tensor, index
        return desc_tensor, scene_img_tensor


