import os
from glob import glob
from glog import logger
from torch.utils.data import Dataset
from data import aug
import torch
from torchvision import transforms
import numpy as np
from PIL import Image

class LRLPRDataset(Dataset):
    def __init__(self, opt, phase):
        super(LRLPRDataset, self).__init__()
        self.opt = opt
        height = opt['height']
        width = opt['width']

        self.dataroot = os.path.join(opt['dataroot'], 'train')
        self.tracks = sorted(glob(os.path.join(self.dataroot, '*', '*', 'track_*')))
            
        self.samples = []
        for track in self.tracks:
            hr_files = sorted(glob(os.path.join(track, 'hr-*.png')))
            
            # Map HR to corresponding LR by index. hr-001 -> lr-001
            for hr_file in hr_files:
                basename = os.path.basename(hr_file) # hr-001.png
                idx_str = basename.split('-')[1].split('.')[0] # 001
                
                # Find corresponding LR
                lr_file = os.path.join(track, f'lr-{idx_str}.png')
                if os.path.exists(lr_file):
                    # Find neighbors if possible
                    idx = int(idx_str)
                    lr_prev = os.path.join(track, f'lr-{idx-1:03d}.png')
                    lr_next = os.path.join(track, f'lr-{idx+1:03d}.png')
                    
                    lrs = []
                    if os.path.exists(lr_prev): lrs.append(lr_prev)
                    lrs.append(lr_file)
                    if os.path.exists(lr_next): lrs.append(lr_next)
                    
                    self.samples.append({
                        'hr': hr_file,
                        'lrs': lrs
                    })

        split_idx = int(len(self.samples) * 0.95)
        if phase == 'train':
            self.samples = self.samples[:split_idx]
        elif phase == 'val':
            self.samples = self.samples[split_idx:]
            
        data_len = opt.get('data_len', -1) if isinstance(opt, dict) else getattr(opt, 'data_len', -1)
        if data_len > 0:
            self.samples = self.samples[:data_len]
                    
        self.transform_fn1 = aug.get_transforms(size=(height, width))
        self.transform_fn2 = aug.get_transforms(size=(height, width))
        self.transform_fn3 = aug.get_transforms(size=(height, width))
        self.transform_fn = aug.get_transforms(size=(height, width))

        self.normalize_fn = aug.get_normalize()
        logger.info(f'Dataset has been created with {len(self.samples)} samples for phase {phase}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        hr_path = sample['hr']
        lrs = sample['lrs']
        
        assert len(lrs) != 0, f'Not enough LR images for index {idx}: {lrs} found, expected at least 1.'
        if len(lrs) < 3:
            sample_id1, sample_id2, sample_id3 = 0, 1, 1
            if len(lrs) < 2:
                sample_id1, sample_id2, sample_id3 = 0, 0, 0
        else:
            sample_id1, sample_id2, sample_id3 = 0, 1, 2
            
        lr_image_1 = Image.open(lrs[sample_id1]).convert('RGB')
        lr_image_2 = Image.open(lrs[sample_id2]).convert('RGB')
        lr_image_3 = Image.open(lrs[sample_id3]).convert('RGB')        
        hr_image = Image.open(hr_path).convert('RGB')
        
        lr_image_1 = np.array(lr_image_1)
        lr_image_2 = np.array(lr_image_2)
        lr_image_3 = np.array(lr_image_3)
        hr_image = np.array(hr_image)

        lr_image_1 = self.transform_fn1(lr_image_1)
        lr_image_2 = self.transform_fn2(lr_image_2)
        lr_image_3 = self.transform_fn3(lr_image_3)
        hr_image = self.transform_fn(hr_image)
        
        lr_image_1 = self.normalize_fn(lr_image_1)
        lr_image_2 = self.normalize_fn(lr_image_2)
        lr_image_3 = self.normalize_fn(lr_image_3)
        hr_image = self.normalize_fn(hr_image)

        lr_image_1 = transforms.ToTensor()(lr_image_1)
        lr_image_2 = transforms.ToTensor()(lr_image_2)
        lr_image_3 = transforms.ToTensor()(lr_image_3)
        hr_image = transforms.ToTensor()(hr_image)

        return {'LR1': lr_image_1, 'LR2': lr_image_2, 'LR3': lr_image_3, 'HR': hr_image, 'path': hr_path} 

    def load_data(self):
        dataloader = torch.utils.data.DataLoader(
            self,
            batch_size=self.opt.batch_size,
            shuffle=True,
            num_workers=int(self.opt.num_threads if hasattr(self.opt, 'num_threads') else 0))
        return dataloader

def create_dataset(opt):
    return LRLPRDataset(opt).load_data()

if __name__ == '__main__':
    class DummyOpt:
        def __init__(self):
            self.dataroot = r'../LRLPR'
            self.mode = 'train'
            self.batch_size = 2
            self.num_threads = 0
            self.height = 112
            self.width = 224
        
        def __getitem__(self, key):
            return getattr(self, key)
            
    opt = DummyOpt()
    dataset = LRLPRDataset(opt, 'val')
    if len(dataset) > 0:
        data = dataset[0]
        print("LR1 shape:", data['LR1'].shape)
        print("LR2 shape:", data['LR2'].shape)
        print("LR3 shape:", data['LR3'].shape)
        print("HR shape:", data['HR'].shape)
        print("Path:", data['path'])
