import copy
import os
import yaml
import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import binary_fill_holes
from data.t2star_loader import (RawMotionT2starDataset,
                                equalize_coil_dimensions,
                                load_segmentations_nii, generate_masks)
from dl_utils.config_utils import set_seed
from utils import *
from utils_plot import *
from mr_utils.parameter_fitting import T2StarFit


""" 
Evaluations performed in this script for MICCAI:

1) MAE, SSIM and FSIM of T2* maps (with  registration to GT)
2) Visual examples of T2* maps and masks
3) Creating figures
"""

set_seed(2109)

parser = argparse.ArgumentParser(description='Evaluate Predictions')
parser.add_argument('--config_path',
                    type=str,
                    default='configs/config_evaluate_bacio.yaml',
                    metavar='C',
                    help='path to configuration yaml file')
args = parser.parse_args()
with open(args.config_path, 'r') as stream_file:
    config = yaml.load(stream_file, Loader=yaml.FullLoader)


if config["load_data_dict"] == "None":
    """ 1. Load input data and predictions """
    print("Loading data from scratch ...")
    data_dict = {
        'sens_maps': {},
        'mask_phimo': {},
        'mask_bootstrap': {},
        'mask_timing': {},
        'img_phimo': {},
        'img_motion': {},
        'img_motion_free': {},
        'img_hrqrmoco': {},
        'img_bootstrap': {},
        'brain_mask': {},
        'gray_matter': {},
        'white_matter': {},
        'slice_indices': {}
    }

    device = 'cuda'
    recon_model = load_recon_model(
        config['recon_model_params'], device=device
    )
    if "bootstrap_recon_model_params" in config.keys():
        bootstrap_recon_model = load_recon_model(
            config['bootstrap_recon_model_params'], device=device
        )
    else:
        bootstrap_recon_model = recon_model

    for subject in config['subjects']:
        print("Processing subject: ", subject)
        Dataset = RawMotionT2starDataset(
            select_one_scan=subject,
            load_whole_set=False,
            **config['data']['params']
        )

        filename_move, filename_gt, _ = Dataset.raw_samples[0]
        slices_ind = sorted(Dataset.get_slice_indices(filename_move))

        for key in data_dict.keys():
            data_dict[key][subject] = []

        bm_loaded = False
        for idx in slices_ind:
            # load images:
            (sens_maps, img_cc_fs,
             img_cc_fs_gt, img_hrqrmoco) = Dataset.load_h5_data(
                filename_move, filename_gt, dataslice=idx,
                load_brainmask=False
            )
            data_dict['slice_indices'][subject].append(idx)
            data_dict['sens_maps'][subject].append(
                equalize_coil_dimensions(sens_maps)
            )
            data_dict['img_motion'][subject].append(img_cc_fs)
            data_dict['img_motion_free'][subject].append(img_cc_fs_gt)
            data_dict['img_hrqrmoco'][subject].append(img_hrqrmoco)

            # load segmentations:
            if not bm_loaded:
                brain_mask, gray_matter, white_matter = load_segmentations_nii(
                    filename_gt
                )
                bm_loaded = True
            data_dict['brain_mask'][subject].append(brain_mask[:, :, idx])
            data_dict['gray_matter'][subject].append(gray_matter[:, :, idx])
            data_dict['white_matter'][subject].append(white_matter[:, :, idx])

        # load predicted masks:
        for idx in slices_ind:
            filename_mask = "{}/{}/{}/predicted_masks/slice_{}.txt".format(
                config['downstream_dir'], config['subjects'][subject],
                subject, idx
            )
            data_dict['mask_phimo'][subject].append(np.loadtxt(filename_mask))

        for key in data_dict.keys():
            if key not in ['img_phimo', 'img_bootstrap', 'mask_bootstrap',
                           'mask_timing']:
                data_dict[key][subject] = np.array(data_dict[key][subject])

        # perform reconstruction:
        for idx in range(0, len(slices_ind)):
            prediction, _ = perform_reconstruction(
                torch.tensor([data_dict['img_motion'][subject][idx]],
                             dtype=torch.complex64, device=device),
                torch.tensor([data_dict['sens_maps'][subject][idx]],
                             dtype=torch.complex64, device=device),
                torch.tensor([data_dict['mask_phimo'][subject][idx]],
                             dtype=torch.float32, device=device),
                recon_model
            )
            data_dict['img_phimo'][subject].append(detach_torch(prediction[0]))

        data_dict['img_phimo'][subject] = abs(
            np.array(data_dict['img_phimo'][subject])
        )
        for key in ['img_motion', 'img_motion_free', 'img_hrqrmoco']:
            data_dict[key][subject] = abs(data_dict[key][subject])

        # bootstrap aggregation results for comparison:
        random_mask_config = config["bootstrap_mask"]
        nr_bootstrap_samples = 15

        for idx in range(0, len(slices_ind)):
            averaged_mask = np.zeros(92)
            averaged_recon = 0
            for i in range(nr_bootstrap_samples):
                bootstrap_mask = generate_masks(random_mask_config,
                                                [1, 1, 92, 1])[0, 0, :, 0]
                prediction, _ = perform_reconstruction(
                    torch.tensor([data_dict['img_motion'][subject][idx]],
                                 dtype=torch.complex64, device=device),
                    torch.tensor([data_dict['sens_maps'][subject][idx]],
                                 dtype=torch.complex64, device=device),
                    torch.tensor([bootstrap_mask],
                                 dtype=torch.float32, device=device),
                    bootstrap_recon_model
                )
                averaged_mask += 1/nr_bootstrap_samples * bootstrap_mask
                averaged_recon += (1/nr_bootstrap_samples *
                                   abs(detach_torch(prediction[0])))
            data_dict['mask_bootstrap'][subject].append(averaged_mask)
            data_dict['img_bootstrap'][subject].append(averaged_recon)
        data_dict['mask_bootstrap'][subject] = np.array(
            data_dict['mask_bootstrap'][subject]
        )
        data_dict['img_bootstrap'][subject] = np.array(
            data_dict['img_bootstrap'][subject]
        )

        if subject == "SQ-struct-00":
            print("Loading the mask from the motion timing "
                         "experiment for comparison for SQ-struct-00.")
            tmp = np.loadtxt(
                "/home/iml/hannah.eichhorn/Data/mqBOLD/RawYoungHealthy"
                "Vol/motion_timing/SQ-struct-00/mask.txt",
                unpack=True).T
            # shift to match the correct timing:
            tmp = np.roll(tmp, 3, axis=1)
            tmp[:, 0:3] = 1
            for idx in slices_ind:
                data_dict['mask_timing'][subject].append(tmp[idx])
            data_dict['mask_timing'][subject] = np.array(
                data_dict['mask_timing'][subject]
            )

    data_dict.pop('sens_maps')


    """ 2. Perform T2star fitting """
    print("Performing T2star fitting ...")
    calc_t2star = T2StarFit(dim=4)
    data_dict['t2star_phimo'] = {}
    data_dict['t2star_motion'] = {}
    data_dict['t2star_motion_free'] = {}
    data_dict['t2star_hrqrmoco'] = {}
    data_dict['t2star_bootstrap'] = {}

    for subject in config['subjects']:
        for img_type in ['phimo', 'motion', 'motion_free', 'hrqrmoco',
                         'bootstrap']:
            data_dict['t2star_{}'.format(img_type)][subject] = detach_torch(
                calc_t2star(
                    torch.tensor(data_dict['img_{}'.format(img_type)][subject]),
                    mask=None,
                )
            )

    # additionally calculate t2star map with last echoes excluded
    data_dict['t2star_phimo_excl'] = {}
    echo_excl_dict = {
        0.05: T2StarFit(dim=4, exclude_last_echoes=2),
        0.1: T2StarFit(dim=4, exclude_last_echoes=3),
        0.15: T2StarFit(dim=4, exclude_last_echoes=4),
        0.2: T2StarFit(dim=4, exclude_last_echoes=5),
        0.25: T2StarFit(dim=4, exclude_last_echoes=6)}

    for subject in config['subjects']:
        img_type = 'phimo'
        calc_t2star_excl = calc_t2star
        excl_rate = (1 - np.sum(data_dict['mask_phimo'][subject]) /
                     np.size(data_dict['mask_phimo'][subject]))
        for key in sorted(echo_excl_dict.keys()):
            if excl_rate >= key:
                calc_t2star_excl = echo_excl_dict[key]

            data_dict['t2star_phimo_excl'.format(img_type)][subject] = detach_torch(
                calc_t2star_excl(
                    torch.tensor(data_dict['img_{}'.format(img_type)][subject]),
                    mask=None,
                )
            )

        # plot echo evolution:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        for idx, mask in enumerate(['gray_matter', 'white_matter']):
            for img_type in ['motion_free', 'phimo', 'motion']:
                if img_type == 'phimo':
                    excl_rate = (1 - np.sum(data_dict['mask_phimo'][subject]) /
                                 np.size(data_dict['mask_phimo'][subject]))
                label = "phimo, excl. rate: {}".format(
                    np.round(excl_rate, 2)) if img_type == 'phimo' \
                    else img_type
                data_plot = np.copy(data_dict['img_{}'.format(img_type)][subject])
                condition = (data_dict[mask][subject] == 0)
                data_plot[condition[:, None].repeat(12, axis=1)] = np.nan
                data_plot_mean = np.nanmean(data_plot, axis=(0, 2, 3))
                axs[idx].plot(data_plot_mean / np.max((data_plot_mean)), '.',
                              label=label)
            axs[idx].legend()
            axs[idx].set_ylim([0.15, 1])
            title = "{}, {}".format(subject, mask)
            axs[idx].set_title(title)
            axs[idx].set_ylabel("Normalised signal intensity")
            axs[idx].set_xlabel("Echo number")
        plt.tight_layout()
        plt.show()


    """ 3. Register to motion-free data """
    # registration to motion-free:
    print("Registering to motion-free data ...")
    data_dict['img_phimo_reg'] = {}
    data_dict['t2star_phimo_reg'] = {}
    data_dict['t2star_phimo_excl_reg'] = {}
    data_dict['img_motion_reg'] = {}
    data_dict['t2star_motion_reg'] = {}
    data_dict['img_hrqrmoco_reg'] = {}
    data_dict['t2star_hrqrmoco_reg'] = {}
    data_dict['img_bootstrap_reg'] = {}
    data_dict['t2star_bootstrap_reg'] = {}

    for subject in config['subjects']:
        for img_type in ['phimo', 'motion', 'hrqrmoco', 'bootstrap']:
            img_reg, t2star_reg, bm_invreg = reg_data_to_gt(
                data_dict['img_motion_free'][subject],
                data_dict['img_{}'.format(img_type)][subject],
                data_dict['t2star_{}'.format(img_type)][subject],
                inplane=False,
                inv_reg=data_dict['brain_mask'][subject]
            )
            data_dict['img_{}_reg'.format(img_type)][subject] = img_reg
            data_dict['t2star_{}_reg'.format(img_type)][subject] = t2star_reg

            # mask the unregistered t2star maps with registered brain mask:
            bm_invreg = np.where(bm_invreg > 0.5, 1, 0)
            filled_bm_reg = []
            for i in range(0, bm_invreg.shape[0]):
                filled_bm_reg.append(binary_fill_holes(bm_invreg[i]))
            filled_bm_reg = np.array(filled_bm_reg)

            data_dict['t2star_{}'.format(img_type)][subject] = (
                data_dict['t2star_{}'.format(img_type)][subject]
                * filled_bm_reg
            )

        # also for t2star_phimo_excl:
        img_reg, t2star_reg, bm_invreg = reg_data_to_gt(
            data_dict['img_motion_free'][subject],
            data_dict['img_phimo'][subject],
            data_dict['t2star_phimo_excl'][subject],
            inplane=False,
            inv_reg=data_dict['brain_mask'][subject]
        )
        data_dict['t2star_phimo_excl_reg'][subject] = t2star_reg

        # mask the unregistered t2star maps with registered brain mask:
        bm_invreg = np.where(bm_invreg > 0.5, 1, 0)
        filled_bm_reg = []
        for i in range(0, bm_invreg.shape[0]):
            filled_bm_reg.append(binary_fill_holes(bm_invreg[i]))
        filled_bm_reg = np.array(filled_bm_reg)

        data_dict['t2star_phimo_excl'][subject] = (
                data_dict['t2star_phimo_excl'][subject]
                * filled_bm_reg
        )

        # mask the motion-free t2star map:
        filled_bm = []
        for i in range(0, data_dict['brain_mask'][subject].shape[0]):
            filled_bm.append(binary_fill_holes(data_dict['brain_mask'][subject][i]))
        data_dict['t2star_motion_free'][subject] = (
            data_dict['t2star_motion_free'][subject] * filled_bm
        )

    # check registered slices for cut-off due to registration and exclude those:
    cutoff_slices = {}
    for subject in config['subjects']:
        cutoff_slices[subject] = []
        for img_type in ['phimo', 'motion', 'hrqrmoco', 'bootstrap']:
             tmp = np.where(
                np.sum(
                    (data_dict['t2star_{}_reg'.format(img_type)][subject] == 0) &
                    (data_dict['t2star_motion_free'][subject] != 0),
                    axis=(1, 2)
                ) > 50
             )[0]
             if len(tmp) > 0:
                cutoff_slices[subject].append(tmp)
        if len(cutoff_slices[subject]) > 0:
            cutoff_slices[subject] = np.unique(
                np.concatenate(cutoff_slices[subject])
            )

    # exclude cutoff slices:
    for subject in config['subjects']:
        for key in data_dict.keys():
            if (subject in data_dict[key].keys()
                    and len(data_dict[key][subject]) > 0):
                data_dict[key][subject] = np.delete(
                    data_dict[key][subject], cutoff_slices[subject], axis=0
                )
        print("Due to registration: excluded slices for "
              "subject {}: {}".format(subject, cutoff_slices[subject]))


    """ Save data_dict: """
    if config["save_data_dict"] != "None":
        if not os.path.exists(os.path.dirname(config["save_data_dict"])):
            os.makedirs(os.path.dirname(config["save_data_dict"]))
        with open(config["save_data_dict"], "wb") as f:
            pickle.dump(data_dict, f)


else:
    """ If config["load_data_dict"] is not "None",
     load the previously saved data_dict. """
    print("Loading processed data_dict from file ...")

    with open(config["load_data_dict"],"rb") as f:
        data_dict = pickle.load(f)


""" 4. Calculate metrics """
print("Calculating metrics ...")
metrics = {
    metric_type: {
        mask: {"stronger": {}, "minor": {}}
        for mask in ["brain_mask", "gray_matter", "white_matter"]}
    for metric_type in ["t2s-MAE", "t2s-SSIM", "t2s-FSIM"]
}

subjects_minor = [s for s in config['motion_severity']
                  if config['motion_severity'][s] == "minor"
                  and s in config['subjects'] ]
subjects_stronger = [s for s in config['motion_severity']
                     if config['motion_severity'][s] == "stronger"
                     and s in config['subjects']]

# T2* metrics:
for mask in ["brain_mask", "gray_matter", "white_matter"]:
    for img_type in ["motion_reg", "phimo_reg", "hrqrmoco_reg",
                     "bootstrap_reg", "phimo_excl_reg"]:
        for metric_type in metrics.keys():
            for motion_type in ["stronger", "minor"]:
                metrics[metric_type][mask][motion_type][img_type] = []

        for subjects, motion_type in zip([subjects_minor, subjects_stronger],
                                         ["minor", "stronger"]):
            for subject in subjects:
                metrics["t2s-MAE"][mask][motion_type][img_type].append(
                    calc_masked_MAE(
                        data_dict["t2star_{}".format(img_type)][subject],
                        data_dict["t2star_motion_free"][subject],
                        detach_torch(data_dict[mask][subject])
                    )
                )
                metrics["t2s-SSIM"][mask][motion_type][img_type].append(
                    calc_masked_SSIM_3D(
                        data_dict["t2star_{}".format(img_type)][subject],
                        data_dict["t2star_motion_free"][subject],
                        detach_torch(data_dict[mask][subject])
                    )
                )
                metrics["t2s-FSIM"][mask][motion_type][img_type].append(
                    calc_masked_FSIM_3D(
                        data_dict["t2star_{}".format(img_type)][subject],
                        data_dict["t2star_motion_free"][subject],
                        detach_torch(data_dict[mask][subject])
                    )
                )

for metric_type in ["t2s-MAE", "t2s-SSIM", "t2s-FSIM"]:
    for mask in ["brain_mask", "gray_matter", "white_matter"]:
        for img_type in ["motion_reg", "phimo_reg", "hrqrmoco_reg",
                         "bootstrap_reg", "phimo_excl_reg"]:
            for motion_type in ["stronger", "minor"]:
                metrics[metric_type][mask][motion_type][img_type] = np.concatenate(
                    metrics[metric_type][mask][motion_type][img_type]
                )

# Statistical Testing:
statistical_tests = {
    metric_type: {
        mask: {
            "stronger": {"combinations": [], "p_values": []},
            "minor": {"combinations": [], "p_values": []}}
        for mask in ["brain_mask", "gray_matter", "white_matter"]}
    for metric_type in ["t2s-MAE", "t2s-SSIM", "t2s-FSIM"]
}

for metric_type in ["t2s-MAE", "t2s-SSIM", "t2s-FSIM"]:
    for mask in ["brain_mask", "gray_matter", "white_matter"]:
        for motion_type in ["stronger", "minor"]:
            print('########################')
            print("Statistical testing of {} for {} and {} motion ...".format(
                metric_type, mask, motion_type))
            combs, p_vals = statistical_testing(
                ["motion_reg",  "bootstrap_reg", "phimo_reg",
                 "hrqrmoco_reg"],
                metrics[metric_type][mask][motion_type]
            )
            statistical_tests[metric_type][mask][motion_type]["combinations"] = combs
            statistical_tests[metric_type][mask][motion_type]["p_values"] = p_vals


# Violin plot for each metric type and mask:
plot_params = {
    "t2s-MAE": {"y_lim": [2, 25],
                "height_brackets": {"stronger": 20, "minor": 12},
                "bracket_at_top": {"stronger": True, "minor": True}},
    "t2s-SSIM": {"y_lim": [0.1, 0.87],
                 "height_brackets": {"stronger": 0.51, "minor": 0.2},
                 "bracket_at_top": {"stronger": True, "minor": False}},
    "t2s-FSIM": {"y_lim": [0.7, 0.98],
                 "height_brackets": {"stronger": 0.8, "minor": 0.81},
                 "bracket_at_top": {"stronger": False, "minor": False}},
}

if "save_folder" in config.keys() and config["save_folder"] != "None":
    save_path = config["save_folder"] + "/figures/violin_"
    create_dir(config["save_folder"] + "/figures/")
else:
    save_path = None

for metric_type, params in plot_params.items():
    for motion_type in ["stronger", "minor"]:
        make_violin_plot(
            metrics,
            statistical_tests,
            img_types_plot=["motion_reg", "bootstrap_reg", "phimo_reg",
                            "hrqrmoco_reg"],
            metric_types_plot=[metric_type],
            motion_types=[motion_type],
            y_lim=params["y_lim"],
            show_title=False,
            height_brackets=params["height_brackets"][motion_type],
            bracket_at_top=params["bracket_at_top"][motion_type],
            plot_legend=False,
            cols=[ "#BEBEBE", "#8497B0", "#005293", "#440154"],
            alphas=[0.8, 0.8, 0.75, 0.75],
            save_path=save_path
        )
# dark blue: #005293, light blue: #8497B0, grey: #A6A6A6, lila: #440154

""" 5. Plot example images """
for subject in config['example_images']:
    if "save_folder" in config.keys() and config["save_folder"] != "None":
        save_path = config["save_folder"] + "/figures/" + subject
        create_dir(config["save_folder"] + "/figures/" + subject)
    else:
        save_path = None

    slices_plot = config['example_images'][subject]
    for s in slices_plot:
        ind = np.where(data_dict['slice_indices'][subject] == s)[0][0]
        echo = 0
        vmin = 0
        vmax = 150
        zoom_coords = [46, 68, 20, 42]

        image_types = ['t2star_motion', 't2star_bootstrap',
                       't2star_phimo',  't2star_hrqrmoco',  't2star_motion_free']

        plt.imshow(np.zeros((20, 20)), cmap='gray')
        plt.text(0.5, 0.5, subject, fontsize=20, color="white",
                 ha="left", va="top")
        plt.text(0.5, 2.5, "Slice {}".format(s), fontsize=20,
                 color="white", ha="left", va="top")
        plt.axis("off")
        plt.show()

        for nr, img_type in enumerate(image_types):
            show_rectangle = True if img_type == 't2star_motion' else False
            colorbar = True if img_type == 't2star_motion_free' else False
            if img_type != 't2star_motion_free':
                mae_gm =  calc_masked_MAE(
                    data_dict[img_type+"_reg"][subject],
                    data_dict["t2star_motion_free"][subject],
                    detach_torch(data_dict["gray_matter"][subject])
                )[ind]
                ssim_gm = calc_masked_SSIM_3D(
                    data_dict[img_type+"_reg"][subject],
                    data_dict["t2star_motion_free"][subject],
                    detach_torch(data_dict["gray_matter"][subject])
                )[ind]
                mae_wm =  calc_masked_MAE(
                    data_dict[img_type+"_reg"][subject],
                    data_dict["t2star_motion_free"][subject],
                    detach_torch(data_dict["white_matter"][subject])
                )[ind]
                ssim_wm = calc_masked_SSIM_3D(
                    data_dict[img_type + "_reg"][subject],
                    data_dict["t2star_motion_free"][subject],
                    detach_torch(data_dict["white_matter"][subject])
                )[ind]
                text = "GM: {} / {}\nWM: {} / {}".format(
                    np.round(mae_gm, 1),
                    np.round(ssim_gm, 2),
                    np.round(mae_wm, 1),
                    np.round(ssim_wm, 2)
                )
            else:
                text = None

            if save_path is not None:
                save_path_img = save_path + "/slice_{}_{}-{}".format(s, nr,
                                                                     img_type)
            else:
                save_path_img = None

            quick_imshow(data_dict[img_type][subject][ind],
                         cmap=None, vmin=vmin,
                         vmax=vmax, colorbar=colorbar,
                         text=text,
                         save_path=save_path_img,
                         zoom_coords=zoom_coords,
                         show_rectangle=show_rectangle
                         )

        excl_masks = ["mask_bootstrap", "mask_phimo"]
        if subject == "SQ-struct-00":
            excl_masks.append("mask_timing")
        for nr, excl_mask in enumerate(excl_masks):
            if save_path is not None:
                save_path_img = save_path + "/mask_slice_{}_{}-{}".format(
                    s, nr, excl_mask.replace("mask_", "")
                )
            else:
                save_path_img = None
            quick_imshow(
                data_dict[excl_mask][subject][ind].reshape(92, 1).repeat(112,
                                                                         axis=1),
                cmap="gray",
                vmin=0,
                vmax=1,
                colorbar=False,
                save_path=save_path_img
            )
            excl_rate = (1 - np.sum(data_dict[excl_mask][subject][ind]) /
                         np.size(data_dict[excl_mask][subject][ind]))
            print(subject, excl_mask, excl_rate)
