import numpy as np
import h5py as h5
import nibabel as nib
import os
import torch
from medutils.mri import ifft2c, rss
from scipy.ndimage import binary_erosion
from scipy.stats import linregress
import datetime


def load_raw_mat_file(file, crop_ro=True):
    """Loading mat files containing raw data converted with MRecon"""

    f = h5.File(file, 'r')
    raw_data = f['out']['Data'][:, :, 0, 0, :, 0]
    tmp = f['out']['Parameter']['YRange'][:]
    if len(np.unique(tmp[0])) > 1 or len(np.unique(tmp[1])) > 1:
        print('Error: different y shifts for different echoes!')
    y_shift = -int((tmp[0,0]+tmp[1,0])/2)

    sens_maps = f['out']['SENSE']['maps'][:, :, 0, 0, :, 0]

    # convert to proper complex data
    if isinstance(raw_data, np.ndarray) and raw_data.dtype == [('real', '<f4'), ('imag', '<f4')]:
        return raw_data.view(np.complex64).astype(np.complex64), y_shift, sens_maps.view(np.complex64).astype(np.complex64)

    else:
        print('Error in load_raw_mat: Unexpected data format: ', raw_data.dtype)


def exponential_decay(t, A, T2star):
    return A * np.exp(-t / T2star)


def t2star_linregr(data, bm, TE1=5, dTE=5):
    """

    :param data: input data to be fitted
    :param bm: brainmask (0 / 1 for background / brain)
    :param TE1: first echo time, default: 5ms
    :param dTE: echo distance, default: 5ms
    :return: fitted T2star and amplitude maps
    """

    TE = np.arange(TE1, data.shape[0] * TE1 + 1, dTE)
    slope, interc = np.zeros(shape=bm.shape), np.zeros(shape=bm.shape)

    fit_data = np.log(data+1e-9)

    for i in range(bm.shape[0]):
        for j in range(bm.shape[1]):
            if bm[i, j]:
                try:
                    result = linregress(TE, fit_data[:, i, j])
                    slope[i, j], interc[i, j] = result.slope, result.intercept
                except:
                    pass

    T2star, A = -1 / slope, np.exp(interc)
    T2star = np.clip(T2star, 0, 200)
    return T2star, A


""" Load example data: """

filename_still = ("/home/iml/hannah.eichhorn/Code/iml-dl/data/links_to_data/"
                  "recon_train_motion_WS/raw_data/SQ-struct-34_nr_"
                  "03072023_1406494_4_2_wip_t2s_air_sg_fV4.mat")

data_still_, y_shift_still, sens_maps_still_ = load_raw_mat_file(filename_still)
dataslice = 10

data_still, sens_maps = data_still_[dataslice], sens_maps_still_[dataslice]
coil_images = ifft2c(data_still)
coil_images = np.roll(coil_images, shift=y_shift_still, axis=-2)
coil_images = coil_images[:, :, :, 56:-56]
sens_maps = np.nan_to_num(sens_maps / rss(sens_maps, 1)[:, None])
img_still = np.sum(coil_images * np.conj(sens_maps), axis=1)
filename_bm_still = filename_still.replace("raw_data", "brain_masks").replace(".mat", "_bm.nii")
bm_still = np.where(nib.load(filename_bm_still).get_fdata()[10:-10][::-1, ::-1, :] < 0.5, 0, 1)[:, :, dataslice]
filename_bm_still_noCSF = os.path.realpath(filename_bm_still).replace("_CSF", "")
bm_still_noCSF = np.where(nib.load(filename_bm_still_noCSF).get_fdata()[10:-10][::-1, ::-1, :] < 0.5, 0, 1)[:, :, dataslice]

# apply binary erosion to the brainmasks to decrease size and remove misregistration errors:
bm_still = binary_erosion(binary_erosion(bm_still))
bm_still_noCSF = binary_erosion(binary_erosion(bm_still_noCSF))


""" Look into differrent fitting methods: """
bm = torch.tensor(bm_still_noCSF)
img = torch.tensor(img_still)
mask = bm > 0
data = torch.log(torch.abs(img[:, mask]) + 1e-9).T


TE1 = 5
dTE = 5
TE = torch.arange(TE1, data.shape[1] * TE1 + 1, dTE, dtype=data.dtype)
TE = TE.unsqueeze(0).repeat(data.shape[0], 1)

# use torch.linalg.lstsq solve AX-B for X with shape (n, k) = (2, 1)
B = data.unsqueeze(-1)   # shape (m, k) = (12, 1)
A = torch.cat((TE.unsqueeze(-1), torch.ones_like(TE).unsqueeze(-1)),
              dim=-1)   # shape (m, n) = (12, 2)
a = datetime.datetime.now()
for i in range(0, 100):
    X = torch.linalg.lstsq(A, B).solution
print("Pytorch duration: ", datetime.datetime.now() - a)

slope, interc = X[:, 0], X[:, 1]
T2star, A0 = -1 / slope, np.exp(interc)
T2star = np.clip(T2star, 0, 200)
print("Pytorch T2star: ", T2star)


# use scipy.stats.linregress
a = datetime.datetime.now()
for i in range(0, data.shape[0]):
    result = linregress(TE[i].numpy(), data[i].numpy())
print("Scipy duration: ", datetime.datetime.now() - a)

slope, interc = result.slope, result.intercept
T2star, A0 = -1 / slope, np.exp(interc)
T2star = np.clip(T2star, 0, 200)
print("Scipy T2star: ", T2star)


""" Comparison Pytorch vs. Scipy: 
Pytorch duration:  0:00:00.805331
Scipy duration:  0:00:00.587970
Pytorch even a little slower than, but comparable and backpropagatable.
"""


print("Done")




img_ = abs(img).unsqueeze(0)
img_ = torch.cat((img_, img_), dim=0)
bm_ = bm.unsqueeze(0)
bm_ = torch.cat((bm_, bm_), dim=0)
bm_ = bm_.unsqueeze(1).repeat(1, 12, 1, 1)

data = img_.reshape(-1, 12)
mask = bm_.reshape(-1, 12)

masked_data = data*mask
masked_data = masked_data[masked_data.sum(dim=1) > 0]


#
# if compare_timing:
#     a = datetime.datetime.now()
#     for i in range(0, 20):
#         T2star, A = t2star_linregr(abs(img_still), bm_still, type="scipy")
#     print("Scipy duration: ", datetime.datetime.now() - a)
#
#     a = datetime.datetime.now()
#     for i in range(0, 20):
#         T2star, A = t2star_linregr(abs(img_still), bm_still, type="torch")
#     print("Torch duration: ", datetime.datetime.now() - a)