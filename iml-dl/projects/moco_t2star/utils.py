import os
import torch
import merlinth
import numpy as np
import wandb
import matplotlib.pyplot as plt
import ants
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from image_similarity_measures.quality_metrics import fsim
from dl_utils.config_utils import import_module
from optim.losses.ln_losses import L2
from optim.losses.image_losses import SSIM_Magn, PSNR_Magn
from optim.losses.physics_losses import T2StarDiff


def create_dir(folder):
    """Create a directory if it does not exist."""

    if not os.path.exists(folder):
        os.makedirs(folder)
    return 0


def detach_torch(data):
    """Detach torch data and convert to numpy."""

    return (data.detach().cpu().numpy()
            if isinstance(data, torch.Tensor) else data)


def process_input_data(device, data):
    """Processes input data for training."""

    img_cc_fs = data[0].to(device)
    sens_maps = data[1].to(device)
    img_cc_fs_gt = data[2].to(device)
    img_hrqrmoco = data[3].to(device)
    brain_mask = data[4].to(device)
    brain_mask_noCSF = data[5].to(device)
    filename = data[6]
    slice_num = data[7].unsqueeze(1).to(torch.float32).to(device)

    return (img_cc_fs, sens_maps, img_cc_fs_gt, img_hrqrmoco, brain_mask,
            brain_mask_noCSF, filename, slice_num)


def round_differentiable(x):
    """Round a tensor in a differentiable way."""

    return x + x.round().detach() - x.detach()


def load_recon_model(recon_dict, device):
    """Load the pretrained reconstruction model."""

    model_class = import_module(recon_dict['module_name'],
                                recon_dict['class_name'])
    recon_model = model_class(**(recon_dict['params']))

    checkpoint = torch.load(recon_dict['weights'],
                            map_location=torch.device(device))
    recon_model.load_state_dict(checkpoint['model_weights'])

    return recon_model.to(device).eval()


def apply_undersampling_mask_torch(img, sens_maps, mask):
    """Apply the given undersampling mask to the given image."""

    coil_imgs = img.unsqueeze(2) * sens_maps
    kspace = merlinth.layers.mri.fft2c(coil_imgs)

    # reshape the mask:
    mask = mask.unsqueeze(1).unsqueeze(1).unsqueeze(-1).repeat(
        1, kspace.shape[1], kspace.shape[2], 1, kspace.shape[-1])

    coil_imgs_zf = merlinth.layers.mri.ifft2c(kspace * mask)
    img_cc_zf = torch.sum(coil_imgs_zf * torch.conj(sens_maps), dim=2)

    A = merlinth.layers.mri.MulticoilForwardOp(
        center=True,
        channel_dim_defined=False
    )
    kspace_zf = A(img_cc_zf, mask, sens_maps)

    return img_cc_zf, kspace_zf, mask.to(img_cc_zf.dtype)


def perform_reconstruction(img, sens_maps, mask_reduced, recon_model):
    """Perform reconstruction using the pretrained reconstruction model.

    Note: If the reconstruction model is a hypernetwork, setup needs to be
    changed and inference needs to be done for each slice individually.
    """

    img_zf, kspace_zf, mask = apply_undersampling_mask_torch(
        img, sens_maps, mask_reduced
    )

    return recon_model(img_zf, kspace_zf, mask, sens_maps), img_zf


def prepare_for_logging(data):
    """Detach the data and crop the third dimension if necessary"""

    data_prepared = detach_torch(data)

    return (data_prepared[:, :, 56:-56] if data_prepared.shape[2] > 112
            else data_prepared)


def convert2wandb(data, abs_max_value, min_value, media_type="video",
                  caption=""):
    """
    Convert normalized data to a format suitable for logging in WandB.

    Parameters
    ----------
    data : np.ndarray
        Input data.
    abs_max_value : np.ndarray
        Maximum absolute value for normalization.
    min_value : float
        Minimum value for normalization.
    media_type : str, optional
        Type of media ("video" or "image"). Default is "video".
    caption : str, optional
        Caption for the logged data. Default is "".

    Returns
    -------
    wandb.Video or wandb.Image
        Formatted data for WandB logging.
    """

    if media_type == "video":
        if np.amin(min_value) < 0:
            return wandb.Video(
                ((np.swapaxes(data[:, None], -2, -1)
                  / abs_max_value[:, None, None, None]+1) * 127.5
                 ).astype(np.uint8),
                fps=0.5, caption=caption
            )
        else:
            return wandb.Video(
                (np.swapaxes(data[:, None], -2, -1)
                 / abs_max_value[:, None, None, None] * 255
                 ).astype(np.uint8),
                fps=0.5, caption=caption
            )
    if media_type == "image":
        if np.amin(min_value) < 0:
            return wandb.Image(
                (np.swapaxes(data[0], -2, -1)
                 / abs_max_value + 1) * 127.5,
                caption=caption
            )
        else:
            return wandb.Image(
                np.swapaxes(data[0], -2, -1) / abs_max_value * 255,
                caption=caption
            )



def log_images_to_wandb(prediction_example, ground_truth_example,
                        motion_example, mask_example, hr_qr_example=None,
                        wandb_log_name="Examples", slice=0,
                        captions=None, data_types=None):
    """Log data to WandB for visualization"""

    if data_types is None:
        data_types = ["magn", "phase"]
    if captions is None:
        captions = ["PHIMO", "Motion-free", "Motion-corrupted"]
    if hr_qr_example is not None:
        captions.append("HR/QR-MoCo")
    data_operations = {
        "magn": np.abs,
        "phase": np.angle,
        "real": np.real,
        "imag": np.imag
    }

    excl_rate = np.round(1 - np.sum(mask_example) / mask_example.size, 2)

    for data_type in data_types:
        pred, gt, motion = map(data_operations[data_type],
                           [prediction_example, ground_truth_example,
                            motion_example])
        if hr_qr_example is not None:
            hr_qr = data_operations[data_type](hr_qr_example)

        # Max / Min values for normalization:
        max_value = np.nanmax(np.abs(np.array([pred, gt, motion])),
                              axis=(0, 2, 3))
        min_value = np.nanmin(np.abs(np.array([pred, gt, motion])),
                              axis=(0, 2, 3))

        # Track multi/single-echo data as video/image data:
        if prediction_example.shape[0] > 1:
            pred = convert2wandb(pred, max_value, min_value,
                                 media_type="video",
                                 caption=captions[0])
            gt = convert2wandb(gt, max_value, min_value,
                               media_type="video",
                               caption=captions[1])
            motion = convert2wandb(motion, max_value, min_value,
                               media_type="video",
                               caption=captions[2])
            mask = np.repeat(mask_example[None, None, :, None], 112, 3)
            mask = wandb.Video(np.swapaxes((mask*255).astype(np.uint8),
                                           -2, -1),
                               fps=0.5,
                               caption="Pred. Mask ({})".format(excl_rate))
            if hr_qr_example is not None:
                hr_qr = convert2wandb(hr_qr, max_value, min_value,
                                        media_type="video",
                                        caption=captions[3])
        else:
            pred = convert2wandb(pred, max_value, min_value,
                                 media_type="image",
                                 caption=captions[0])
            gt = convert2wandb(gt, max_value, min_value,
                               media_type="image", caption=captions[1])
            motion = convert2wandb(motion, max_value, min_value,
                               media_type="image", caption=captions[2])
            mask = np.repeat(mask_example[:, None], 112, 1)
            mask = wandb.Image(np.swapaxes((mask*255).astype(np.uint8),
                                           -2, -1),
                               caption="Pred. Mask ({})".format(excl_rate))
            if hr_qr_example is not None:
                hr_qr = convert2wandb(hr_qr, max_value, min_value,
                                        media_type="image",
                                        caption=captions[3])

        log_key = "{}{}/slice_{}".format(wandb_log_name, data_type, slice)
        log_data = [motion, pred, gt, mask]
        if hr_qr_example is not None:
            log_data.append(hr_qr)
        wandb.log({log_key: log_data})


def determine_echoes_to_exclude(mask):
    """Determine the number of echoes to exclude based on the mask."""

    exclusion_rate = 1 - torch.sum(mask) / mask.numel()

    leave_last_dict = {0.08: 2, 0.18: 3, 0.28: 4}
    leave_last = None

    for key in sorted(leave_last_dict.keys()):
        if exclusion_rate >= key:
            leave_last = leave_last_dict[key]

    return leave_last


def log_t2star_maps_to_wandb(t2star_pred, t2star_gt,
                             t2star_motion, t2star_hrqr=None,
                             wandb_log_name="Examples", slice=0):
    """Log data to WandB for visualization"""

    figure_size = (5, 2.5)
    t2stars = [t2star_motion, t2star_pred, t2star_gt]
    titles = ["Motion-corrupted", "PHIMO", "Motion-free"]

    if t2star_hrqr is not None:
        figure_size = (6.5, 2.5)
        t2stars.append(t2star_hrqr)
        titles.append("HR/QR-MoCo")

    fig = plt.figure(figsize=figure_size, dpi=300)
    min_value = 0
    max_value = 200

    for nr, (map, title) in enumerate(zip(t2stars, titles)):
        plt.subplot(1, len(titles)+1, nr + 1)
        plt.imshow(map.T, vmin=min_value, vmax=max_value)
        plt.axis("off")
        plt.title(title, fontsize=8)
        if nr == len(titles)-1:
            cax = plt.axes([0.75, 0.3, 0.025, 0.35])
            cbar = plt.colorbar(cax=cax)
            cbar.ax.tick_params(labelsize=8)

    log_key = "{}slice_{}".format(wandb_log_name, slice)
    wandb.log({log_key: fig})


def calculate_img_metrics(target, data, bm, metrics_to_be_calc,
                          include_brainmask=True):
    """ Calculate metrics for a given target array and data array."""

    metrics = {}
    methods_dict = {
        'MSE': L2,
        'SSIM': SSIM_Magn,
        'PSNR': PSNR_Magn
    }

    for descr in metrics_to_be_calc:
        for m in methods_dict:
            if descr.startswith(m):
                metric = methods_dict[m](include_brainmask)
                break
        if "magn" in descr:
            metrics[descr] = metric(
                torch.abs(target), torch.abs(data),
                bm
            ).item()
        elif "phase" in descr:
            metrics[descr] = metric(
                torch.angle(target), torch.angle(data),
                bm
            ).item()

    return metrics

def calculate_t2star_metrics(target, data, bm, metrics_to_be_calc):

    metrics = {}
    methods_dict = {
        'T2s_MAE': T2StarDiff
    }

    for descr in metrics_to_be_calc:
        for m in methods_dict:
            if descr.startswith(m):
                metric = methods_dict[m]()
                break
        metrics[descr] = metric(
            target, data, bm
        ).item()

    return metrics


def rigid_registration(fixed, moving, *images_to_move,
                       inv_reg=None, numpy=False, inplane=True):
    """Perform rigid registration of moving to fixed image."""

    if not numpy:
        device = fixed.device
        # convert to numpy:
        fixed = detach_torch(fixed)
        moving = detach_torch(moving)
        images_to_move = [detach_torch(im) if not isinstance(im, np.ndarray)
                          else im for im in images_to_move]

    # calculate transform for fixed and moving
    fixed_image = ants.from_numpy(abs(fixed))
    moving_image = ants.from_numpy(abs(moving))

    # Perform registration
    registration_result = ants.registration(
        fixed=fixed_image,
        moving=moving_image,
        type_of_transform="Rigid",
        random_seed=2019
    )

    # apply it to the other images
    images_reg = []
    if inplane:
        for im in images_to_move:
            im_reg = np.zeros_like(im)
            for i in range(len(im)):
                ants_image = ants.from_numpy(im[i])

                # Apply the transformation to the image
                im_reg[i] = ants.apply_transforms(
                    fixed_image,
                    moving=ants_image,
                    transformlist=registration_result['fwdtransforms']
                ).numpy()
            images_reg.append(im_reg)
    else:
        for im in images_to_move:
            im_reg = np.zeros_like(im)
            for i in range(im.shape[1]):
                ants_image = ants.from_numpy(im[:, i])

                # Apply the transformation to the image
                im_reg[:, i] = ants.apply_transforms(
                    fixed_image,
                    moving=ants_image,
                    transformlist=registration_result['fwdtransforms']
                ).numpy()
            images_reg.append(im_reg)

    if inv_reg is not None:
        if inplane:
            inv_reg_ = []
            for i in range(len(inv_reg)):
                ants_image = ants.from_numpy(inv_reg[i].astype(float))
                # Apply the transformation to the mask
                inv_reg_slice = ants.apply_transforms(
                    fixed_image,
                    moving=ants_image,
                    transformlist=registration_result['invtransforms']
                ).numpy()
                inv_reg_.append(inv_reg_slice)
        else:
            ants_image = ants.from_numpy(inv_reg.astype(float))
            # Apply the transformation to the mask
            inv_reg_ = ants.apply_transforms(
                fixed_image,
                moving=ants_image,
                transformlist=registration_result['invtransforms']
            ).numpy()

        if not numpy:
            output = [torch.tensor(im).to(device) for im in images_reg].append(
                torch.tensor(inv_reg_).to(device))
            return output
        else:
            images_reg.append(inv_reg_)
            return images_reg
    else:
            if not numpy:
                return [torch.tensor(im).to(device) for im in images_reg]
            else:
                return images_reg


def reg_data_to_gt(img_gt, img, t2star,
                   inplane=True,  inv_reg=None):
    """Register image and T2star map to ground truth image."""

    if inplane:
        img_reg, t2star_reg, inv_reg_ = [], [], []
        for i in range(len(img_gt)):
            reg_result = rigid_registration(abs(img_gt[i])[0],
                                            abs(img[i])[0],
                                            abs(img)[i],
                                            [t2star[i]],
                                            inv_reg=inv_reg,
                                            numpy=True,
                                            inplane=inplane)
            img_reg.append(reg_result[0])
            t2star_reg.append(reg_result[1][0])
            if inv_reg is not None:
                inv_reg_ .append(reg_result[2])
    else:
       reg_result = rigid_registration(
           abs(img_gt)[:, 0],
           abs(img)[:, 0],
           abs(img),
           t2star[:, None],
           inv_reg=inv_reg,
           numpy=True,
           inplane=inplane
       )
       img_reg = reg_result[0]
       t2star_reg = reg_result[1][:, 0]
       if inv_reg is not None:
           inv_reg_ = reg_result[2]

    if inv_reg is None:
        return np.array(img_reg), np.array(t2star_reg)
    else:
        return np.array(img_reg), np.array(t2star_reg), np.array(inv_reg_)


def calc_masked_MAE(img1, img2, mask):
    """Calculate Mean Absolute Error between two images in a specified mask"""

    masked_diff = np.ma.masked_array(abs(img1 - img2),
                                     mask=(mask[:, None] != 1))
    return np.mean(masked_diff, axis=(1, 2)).filled(0)


def calc_masked_SSIM_3D(img, img_ref,  mask):
    """Calculate SSIM between two 3D images in a specified mask"""

    ssims = []
    for i in range(len(img)):
        mssim, ssim_values = structural_similarity(
            img_ref[i], img[i], data_range=np.amax(img_ref[i]),
            gaussian_weights=True, full=True
        )
        masked = np.ma.masked_array(ssim_values, mask=(mask[i] != 1))
        ssims.append(np.mean(masked))
    return np.array(ssims)


def calc_masked_FSIM_3D(img, img_ref, mask):
    """Calculate FSIM between two 3D images in a specified mask

    Note: The masked is multiplied to the images and not the FSIM values.
    """

    img = img * mask
    img_ref = img_ref * mask
    fsims = []
    for i in range(len(img)):
        fsims.append(fsim(org_img=img_ref[i][:,  :, None],
                          pred_img=img[i][:,  :, None]))
    return np.array(fsims)


def calc_masked_SSIM_4D(img, img_ref, mask, av_echoes=True,
                        later_echoes=False, normalize=True):
    """Calculate SSIM between two 4D images in a specified mask"""

    ssims = []
    for i in range(img.shape[0]):
        if normalize:
            ref = (img_ref[i] - np.mean(img_ref[i])) / np.std(img_ref[i])
            data = (img[i] - np.mean(img[i])) / np.std(img[i])
        else:
            ref = img_ref[i]
            data = img[i]

        ssim_echoes = []
        for j in range(data.shape[0]):
            mssim, ssim_values = structural_similarity(
                ref[j], data[j], data_range=np.amax(ref[j]),
                gaussian_weights=True, full=True
            )
            masked = np.ma.masked_array(ssim_values, mask=(mask[i] != 1))
            ssim_echoes.append(np.mean(masked))
        if av_echoes:
            ssims.append(np.mean(ssim_echoes))
        elif later_echoes:
            ssims.append(np.mean(ssim_echoes[-later_echoes:]))
        else:
            ssims.append(ssim_echoes)
    return np.array(ssims)


def calc_masked_PSNR_4D(img, img_ref, mask, av_echoes=True,
                        later_echoes=False, normalize=True):
    """Calculate PSNR between two 4D images in a specified mask"""

    psnrs = []
    for i in range(img.shape[0]):
        psnr_echoes = []
        if normalize:
            ref = (img_ref[i] - np.mean(img_ref[i])) / np.std(img_ref[i])
            data = (img[i] - np.mean(img[i])) / np.std(img[i])
        else:
            ref = img_ref[i]
            data = img[i]

        for j in range(data.shape[0]):
            d_ref = ref[j].flatten()[mask[i].flatten() > 0]
            d_img = data[j].flatten()[mask[i].flatten() > 0]
            psnr_echoes.append(
                peak_signal_noise_ratio(d_ref, d_img,
                                        data_range=np.amax(ref[j]))
            )
        if av_echoes:
            psnrs.append(np.mean(psnr_echoes))
        elif later_echoes:
            psnrs.append(np.mean(psnr_echoes[-later_echoes:]))
        else:
            psnrs.append(psnr_echoes)
    return np.array(psnrs)


def calc_masked_FSIM_4D(img, img_ref, mask, av_echoes=True,
                        later_echoes=False, normalize=True):
    """Calculate FSIM between two 4D images in a specified mask"""

    fsims = []
    for i in range(img.shape[0]):
        fsim_echoes = []
        if normalize:
            ref = (img_ref[i] - np.mean(img_ref[i])) / np.std(img_ref[i])
            data = (img[i] - np.mean(img[i])) / np.std(img[i])
        else:
            ref = img_ref[i]
            data = img[i]

        data = data * mask[i]
        ref = ref * mask[i]

        for j in range(data.shape[0]):
            fsim_echoes.append(fsim(org_img=ref[j][:, :, None],
                                    pred_img=data[j][:, :, None]))
        if av_echoes:
            fsims.append(np.mean(fsim_echoes))
        elif later_echoes:
            fsims.append(np.mean(fsim_echoes[-later_echoes:]))
        else:
            fsims.append(fsim_echoes)
    return np.array(fsims)
