"""
Script for transforming the motion timing instructions for the motion
timing experiment into a corresponding exclusion mask.
"""

import numpy as np


scan_order_file = ("/path_to_file_with_scan_acquisition_scheme/"
                   "Scan_order_36.txt")

time, _, echo, sl, pe = np.loadtxt(scan_order_file, unpack=True)

start_timing = [5, 50, 105, 150, 165, 195]
end_timing = [10, 60, 110, 155, 170, 205]

mask = np.ones_like(time)
for s, e in zip(start_timing, end_timing):
    mask[(time >= s) & (time < e)] = 0

mask = np.round(mask.reshape(-1, 12).mean(axis=1))
sl = sl[::12]
pe = pe[::12]

# Create a structured array with 'sl' and 'pe' as fields
structured_array = np.array(list(zip(sl, pe, mask)),
                            dtype=[('sl', 'f8'), ('pe', 'f8'), ('mask', 'f8')])

# Sort by 'sl' first, then by 'pe'
sorted_array = np.sort(structured_array, order=['sl', 'pe'])

# Extract the sorted 'sl', 'pe', and 'mask' arrays
sl_sorted = sorted_array['sl']
pe_sorted = sorted_array['pe']
mask_sorted = sorted_array['mask']

final_mask = mask_sorted.reshape(-1, 92)
np.savetxt("/path_to_save_result/mask.txt",
           final_mask, fmt='%d')
