import numpy as np
import cv2  # OpenCV is used for Gaussian blur
import rasterio
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision

from utils import load_fun
from scipy.ndimage import zoom, rotate
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# Downsampling factor
downsample_factor = 4  # Set as needed

'''
    Given an image I_c and I_s the goal is to find matrix A such that:
              Cov(AI_c) = Cov(I_s)
              µ(AI_c)   = µ(I_s)

    Whitening transform produces I' from I_c with Cov(I') = Unit.
              I' = W I_c
              W  = Λ_c^{−1/2} Φ_c^T I_c
              with Cov(I_c) = Φ_c Λ_c^{1/2} Λ_c^{1/2} Φ_c^T

    Colouring transform transforms unit random variable I' to random variable with desired Cov. 
            Cov(C I') = Cov(I_s)
            µ(C I')   = Cov(I_s)
    with 
                   C  = Φ_s Λ_s^{1/2} I'

    Now we also need to adjust the mean values. For this we need to first move the I_c by the mean µ_c

                  I'' = C(W I_c - µ_c) I_c + µ_s.
                  
    Further reading: https://www.projectrhea.org/rhea/index.php/ECE662_Whitening_and_Coloring_Transforms_S14_MH

'''


'''
    Code and method based on: https://github.com/AronKovacs/g-style
'''
def match_color(input_image, gt_image):
    """
    input_img: [N, 3, H, W]
    gt_image: [N, 3, H, W]
    """
    gt_image = gt_image.permute(0, 2, 3, 1)
    input_image = input_image.permute(0, 2, 3, 1)
    batch_size, height, width, nc = gt_image.shape
    sh                            = gt_image.shape
    gt_image = gt_image.view(batch_size, -1, 3)
    input_image = input_image.view(batch_size, -1, 3).to(gt_image.device)
    # Mean values
    mu_c = gt_image.mean(1, keepdim=True)
    mu_s = input_image.mean(1, keepdim=True)
    """
        image_set: [N, H, W, 3]
        style_img: [H, W, 3]
    """
    # Covariance
    cov_c = torch.matmul((gt_image - mu_c).transpose(2, 1), gt_image - mu_c) / float(gt_image.size(1))
    cov_s = torch.matmul((input_image - mu_s).transpose(2, 1), input_image - mu_s) / float(input_image.size(1))
    # N, 3, 3
    # Cov(I_c) = Φ_c Λ_c^{1/2} Λ_c^{1/2} Φ_c^T
    u_c, sig_c, v_c = torch.svd(cov_c)
    # Cov(I_s) = Φ_s Λ_s^{1/2} Λ_s^{1/2} Φ_s^T
    u_s, sig_s, v_s = torch.svd(cov_s)

    # Φ_c^T
    u_c_i = u_c.transpose(2, 1)
    # Φ_s^T
    u_s_i = u_s.transpose(2, 1)


    # Inverse of second component of SVD
    # Λ_c^{−1/2}
    scl_c = torch.diag_embed(1.0 / torch.sqrt(torch.clamp(sig_c, 1e-8, 1e8)))
    # Λ_s^{1/2}
    scl_s = torch.diag_embed(torch.sqrt(torch.clamp(sig_s, 1e-8, 1e8)))

    tmp_mat = u_s @ scl_s @ u_s_i @ u_c @ scl_c @ u_c_i
    tmp_vec = mu_s - mu_c @ tmp_mat.transpose(2, 1)

    gt_image = gt_image @ tmp_mat.transpose(2, 1) + tmp_vec
    gt_image = gt_image.contiguous().clamp_(0.0, 1.0).view(sh)

    ''' 
    TODO: do we need this?
    color_tf = torch.eye(4).float().to(tmp_mat.device)
    color_tf[:3, :3] = tmp_mat
    color_tf[:3, 3:4] = tmp_vec.T
    '''
    return gt_image.permute(0, 3, 1, 2) # [B, C, H, W]


def normalize_sentinel_13band(sentinel_image, GTMean, GTStd):
    
    height, width = sentinel_image.shape[1], sentinel_image.shape[2]
    normalized_rgb = np.zeros((13, height, width))

    sentinel_rgb = np.stack([sentinel_image[3, :, :],  # Red (Band 4)
                             sentinel_image[2, :, :],  # Green (Band 3)
                             sentinel_image[1, :, :],
                             sentinel_image[7, :, :],
                             sentinel_image[4, :, :],
                             sentinel_image[5, :, :],
                             sentinel_image[6, :, :],
                             sentinel_image[8, :, :],
                             sentinel_image[11, :, :],
                             sentinel_image[12, :, :],
                             sentinel_image[0, :, :],
                             sentinel_image[9, :, :],
                             sentinel_image[10, :, :]         
                             ], axis=0)  # Blue (Band 2)

    global_mean = np.array(GTMean)  # [mean_R, mean_G, mean_B]
    global_std = np.array(GTStd)  # [std_R, std_G, std_B]
    global_s2_mean = np.array([761.8781, 920.9624, 972.6395,3037.5247,1114.0452, 2439.3484, 3075.5315,3409.9683,  1856.8668, 995.5594,1211.6917, 1015.3680, 12.0481])
    global_s2_std = np.array([417.8611, 270.1484, 220.4282, 718.2054,408.2447, 498.0920, 677.2336,743.4434,616.0681, 506.1452, 146.7525, 191.8340, 2.0941])
    # Normalize each RGB channel
    for i in range(13):  # Loop over R, G, B, VNIR bands
        if i >= 3:
            normalized_rgb[i, :, :] = (sentinel_rgb[i, :, :] - global_s2_mean[i]) / global_s2_std[i]
        else:
            normalized_rgb[i, :, :] = (sentinel_rgb[i, :, :] - global_s2_mean[i]) / global_s2_std[i] * global_std[i] + global_mean[i]

    normalized_bands = np.clip(normalized_rgb, 0, 255).astype(np.uint8)

    return normalized_bands


def resize_np_file(file_path):
    # Load the .npy file
    data = np.load(file_path)
    # global_mean = [87.9528, 99.8053, 91.6490]  # Provided global mean for R, G, B channels
    # global_std = [36.1209, 41.8382, 50.4368]  # Provided global std for R, G, B channels
    # normalized_image = normalize_sentinel_13band(data, global_mean, global_std)

    
    # # Check if the data has the correct shape
    # if normalized_image.shape != (13, 100, 100):
    #     raise ValueError(f"Expected data shape (10, 100, 100), but got {normalized_image.shape}")

    # # Reshape the data to (13, 100, 100) if necessary (ensure it is in channel-first format)
    # channels, height, width = normalized_image.shape

    # # Resize each channel to (400, 400)
    # zoom_factors = (1, 120 / height, 120 / width)
    # resized_data = zoom(normalized_image, zoom_factors, order=3)  # order=3 for cubic interpolation
    # return resized_data
    return data


def load_file(filename):
    with rasterio.open(filename) as src:
        file_ = src.read()[:3]  # Read only the first three channels (R, G, B)
    return file_


def dummy_load_image(s2_path, dop20_path):
    rbg_s2_means = torch.tensor([972.6395, 920.9624, 761.8781])
    s2_image = np.load(s2_path)
    # Open the image
    dop20_image = Image.open(dop20_path)

    # Define the transform to resize the image to 100x100
    transform = transforms.Compose([
        transforms.Resize((100, 100)),  # Resize to 100x100
        transforms.ToTensor(),          # Convert to tensor
    ])

    # Apply the transformation
    dop20_image = transform(dop20_image).float()
    s2_image = torch.from_numpy(s2_image[[3, 2, 1], :, :]).float()
    s2_image_max = s2_image.amax(dim=(1, 2))
    s2_image[0] /= s2_image_max[0]
    s2_image[1] /= s2_image_max[1]
    s2_image[2] /= s2_image_max[2]
    s2_image = torchvision.transforms.Resize(dop20_image.shape[1:], antialias=True)(s2_image)
    dop20_image = dop20_image.unsqueeze(0)
    s2_image = s2_image.unsqueeze(0)

    return s2_image, dop20_image

def load_file_np(filenamelr, filenamehr, isColorMatching=False):
    # file_ = resize_np_file(filename)
    # file_ = np.load(filename)
    # global_mean = [87.9528, 99.8053, 91.6490]  # Provided global mean for R, G, B channels
    # global_std = [36.1209, 41.8382, 50.4368]  # Provided global std for R, G, B channels
    # normalized_image = normalize_sentinel_13band(file_, global_mean, global_std)
    if isColorMatching:
        s2_image_0, dop20_image_0 = dummy_load_image(filenamelr, filenamehr)
        s2_to_dop = match_color(s2_image_0, dop20_image_0)
        dop_to_s2 = match_color(dop20_image_0, s2_image_0)
 
        return dop_to_s2[0].numpy()
    else:
        sentinel_image = np.load(filenamelr)
        # sentinel_rgb = np.stack([sentinel_image[3, :, :],  # Red (Band 4)
        #                     sentinel_image[2, :, :],  # Green (Band 3)
        #                      sentinel_image[1, :, :]]
        #                      , axis=0
        # )
        # sentinel_rgb = np.clip(sentinel_rgb, 0, 255).astype(np.uint8)
        # # Convert the sentinel_rgb array to float before dividing
        # sentinel_rgb = sentinel_rgb.astype(np.float32)  # or np.float64

        # # Now perform the division
        # sentinel_rgb = normalize_sentinel_13band(sentinel_image)
        # sentinel_rgb = sentinel_rgb / 255.0
        data = np.load(filenamelr)
        global_mean = [87.9528, 99.8053, 91.6490]  # Provided global mean for R, G, B channels
        global_std = [36.1209, 41.8382, 50.4368]  # Provided global std for R, G, B channels
        normalized_image = normalize_sentinel_13band(data, global_mean, global_std)
        normalized_image = normalized_image / 255.0


        return normalized_image

def _load_files_dir(fdir):
    print('load files from {}'.format(fdir))
    file_list = []
    for dirpath, dirs, files in os.walk(fdir):
        for filename in files:
            if filename.endswith(('.jpg', '.jpeg', '.npy', '.png')):
                file_list.append(os.path.join(dirpath, filename))
    return sorted(file_list)

# class OLI2MSI(Dataset):
#     def __init__(self, cfg, is_training=True):
#         print('dataset OLI2MSI')
#         self.root_path = cfg.dataset.root_path
#         self.relname = 'train'
#         if not is_training:
#             self.relname = 'test'
#         self.fdir_lr = os.path.join(self.root_path, self.relname + '_lr')
#         self.fdir_hr = os.path.join(self.root_path, self.relname + '_hr')
#         self.apply_transformations = True
#         self._load_files()

#     def _load_files(self):
#         self.files_lr = _load_files_dir(self.fdir_lr)
#         self.files_hr = _load_files_dir(self.fdir_hr)

#     def __len__(self):
#         return len(self.files_lr)

#     def __getitem__(self, index):
#         # Apply transformations to LR and HR images
#         # file_lr = load_file_np(self.files_lr[index])
#         # file_lr = load_file_np(self.files_lr[index])
#         # if self.relname == 'test':
#         #     file_lr = load_file_np(self.files_lr[index], self.files_hr[index], False)
#         # else:
#         #     file_lr = load_file_np(self.files_lr[index], self.files_hr[index], True)

#         file_lr = resize_np_file(self.files_lr[index])
#         file_hr = load_file(self.files_hr[index])
        
#         # Normalize the images by dividing by 255
#         # file_lr = file_lr / 255.0
#         file_hr = file_hr / 255.0
        
#         return file_lr, file_hr

class OLI2MSI(Dataset):
    def __init__(self, cfg, is_training=True):
        print('dataset OLI2MSI')
        self.root_path = cfg.dataset.root_path
        self.relname = 'train' if is_training else 'test'
        self.fdir_lr = os.path.join(self.root_path, f'{self.relname}_lr')
        self.fdir_hr = os.path.join(self.root_path, f'{self.relname}_hr')
        self.apply_transformations = True
        self._load_files()

    def _load_files(self):
        self.files_lr = _load_files_dir(self.fdir_lr)
        self.files_hr = _load_files_dir(self.fdir_hr)

    def __len__(self):
        return len(self.files_lr)

    def __getitem__(self, index):
        # Load images and normalize
        file_lr = resize_np_file(self.files_lr[index])
        file_hr = load_file(self.files_hr[index])
        file_name = os.path.basename(self.files_hr[index]).split('.')[0]

        # Normalize images by dividing by 255
        file_hr = file_hr / 255.0
        
        return {'lr': file_lr, 'hr': file_hr, 'file_name': file_name}


def load_dataset(cfg, only_test=False, concat_datasets=False):
    collate_fn = cfg.dataset.get('collate_fn')
    if collate_fn is not None:
        collate_fn = load_fun(collate_fn)

    persistent_workers = False
    if cfg.num_workers > 0:
        persistent_workers = True

    train_dset = None
    train_dloader = None
    concat_dloader = None

    if concat_datasets:
        train_dset = OLI2MSI(cfg, is_training=True)
        val_dset = OLI2MSI(cfg, is_training=False)
        dset = torch.utils.data.ConcatDataset([train_dset, val_dset])

        shuffle = True

        concat_dloader = DataLoader(
            dset,
            batch_size=cfg.batch_size,
            sampler=None,
            shuffle=shuffle,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=persistent_workers,
            collate_fn=collate_fn(cfg)
        )

    if not only_test:
        train_dset = OLI2MSI(cfg, is_training=True)

        train_dloader = DataLoader(
            train_dset,
            batch_size=cfg.batch_size,
            sampler=None,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=persistent_workers,
            collate_fn=collate_fn(cfg),
        )

    val_dset = OLI2MSI(cfg, is_training=False)

    shuffle = False

    # TODO distribute also val_dset
    val_dloader = DataLoader(
        val_dset,
        batch_size=cfg.batch_size,
        sampler=None,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn(cfg)
    )

    return train_dloader, val_dloader, concat_dloader
