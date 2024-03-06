import numpy as np
from skimage.morphology import disk, ball, closing, binary_dilation, dilation, binary_erosion
from infer_subc.core.img import fill_and_filter_linear_size, get_max_label, inverse_log_transform, log_transform, masked_inverted_watershed, select_channel_from_raw, threshold_otsu_log
from aicssegmentation.core.utils import hole_filling
from infer_subc.constants import (
    TEST_IMG_N,
    NUC_CH,
    LYSO_CH,
    MITO_CH,
    GOLGI_CH,
    PEROX_CH,
    ER_CH,
    LD_CH,
    PM_CH,
    RESIDUAL_CH,
)

def rescale_intensity(in_img: np.ndarray):
    #rescales the intensity of input image on a scale of 0 to 10
    out_img = ((in_img - in_img.min())/(in_img.max() - in_img.min()))*10
    return out_img

def membrane_composite(in_img: np.ndarray,
                       weight_ch0: int = 0,
                       weight_ch1: int = 0,
                       weight_ch2: int = 0,
                       weight_ch3: int = 0,
                       weight_ch4: int = 0,
                       weight_ch5: int = 0,
                       weight_ch6: int = 0,
                       weight_ch7: int = 0,
                       weight_ch8: int = 0,
                       weight_ch9: int = 0,
                       Invert_PM: bool = False,
                       PM_Channel: int = 0,
                     ):
    weights = [weight_ch0,
               weight_ch1,
               weight_ch2,
               weight_ch3,
               weight_ch4,
               weight_ch5,
               weight_ch6,
               weight_ch7,
               weight_ch8,
               weight_ch9]
    raw_img = in_img.copy()
    out_img = np.zeros_like(in_img[0]).astype(np.double)
    if Invert_PM:
        raw_img[PM_Channel] = abs(np.max(in_img[PM_Channel])-in_img[PM_Channel])
    for ch, weight in enumerate(weights):
        if weight > 0:
            out_img+=weight*rescale_intensity(raw_img[ch])
    return out_img

def close(in_img: np.ndarray, 
          Method: str, 
          Size: int):
    out_img = np.zeros_like(in_img)
    if Method == "Ball":
        if Size > 0:
            fp = ball(Size)
            out_img = closing(in_img.copy(), footprint=fp)
        else:
            out_img = closing(in_img.copy())
    elif Method == "Disk":
        if Size > 0:
            fp = disk(Size)
            for z in range(len(in_img)):
                out_img[z] = closing(in_img[z].copy(), footprint=fp)
        else:
            for z in range(len(in_img)):
                out_img[z] = closing(in_img[z].copy())
    return out_img

def find_nuc(nuc_img: np.ndarray, 
             in_img: np.ndarray):
    keep_nuc = get_max_label((in_img), dilation(nuc_img))
    out_img = np.zeros_like(nuc_img)
    out_img[nuc_img == keep_nuc] = 1
    return out_img

def mix_nuc(single_nuc: np.ndarray, 
            in_img: np.ndarray, 
            Method: str, 
            Size: int):
    out_img = in_img.copy()
    if Method == "Disk":
        if Size > 0:    
            nuc_fp = disk(Size)
            for z in range(len(in_img)):
                out_img[z] += binary_dilation(single_nuc.astype(bool)[z], footprint=nuc_fp)
        else:
            for z in range(len(in_img)):
                out_img[z] += binary_dilation(single_nuc.astype(bool)[z])
    elif Method == "Ball":
        if Size > 0:  
            nuc_fp = ball(Size)
            out_img += binary_dilation(single_nuc.astype(bool), footprint=nuc_fp)
        else:
            out_img += binary_dilation(single_nuc.astype(bool))
    return out_img

def fill_and_bind(raw_img: np.ndarray, 
                  in_img: np.ndarray, 
                  Min_Hole_Width: int, 
                  Max_Hole_Width: int,
                  Small_Obj_Width: int,
                  PM_Channel: int,
                  Thresh_Adj: float,
                  Bind_to_PM: bool):
    filled = fill_and_filter_linear_size(in_img,
                                         hole_min=Min_Hole_Width,
                                         hole_max=Max_Hole_Width,
                                         min_size=Small_Obj_Width)
    if Bind_to_PM:
        pm_img = select_channel_from_raw(raw_img, PM_Channel)
        pm_image, d = log_transform(pm_img.copy())
        pm_thresh = threshold_otsu_log(pm_image) 
        invert_pm_obj = np.invert(pm_img > (inverse_log_transform(pm_thresh, d) * Thresh_Adj))
        mask = np.zeros_like(invert_pm_obj)
        mask[(invert_pm_obj == filled) & 
                 (invert_pm_obj == 1) & 
                 (filled == 1)] = 1
        out_img = closing(mask)
    else:
        out_img = filled.copy()
    return out_img

def double_watershed(single_nuc: np.ndarray,
                     raw_img_A: np.ndarray,
                     raw_img_B: np.ndarray,
                     thresh_img_A: np.ndarray,
                     thresh_img_B: np.ndarray,
                     Watershed_Method: str="3D"):
    cm_A = masked_inverted_watershed(raw_img_A, 
                                     single_nuc, 
                                     thresh_img_A,
                                     method=Watershed_Method)
    cm_B = masked_inverted_watershed(raw_img_B, 
                                     single_nuc, 
                                     thresh_img_B,
                                     method=Watershed_Method)
    cm_combo = cm_A.astype(bool) + cm_B.astype(bool)
    return cm_combo

def close_and_fill(in_img: np.ndarray, 
                   Min_Hole_Width: int,
                   Max_Hole_Width: int,
                   Method: str,
                   Size: int):
    if Method == "Disk":
        if Size > 0:
            close_fp = disk(Size)
            for z in range(len(in_img)):
                in_img[z] = binary_dilation(in_img.copy()[z], footprint=close_fp)
        else:
            for z in range(len(in_img)):
                in_img[z] = binary_dilation(in_img.copy()[z])
    elif Method == "Ball":
        if Size > 0:
            close_fp = ball(Size)
            in_img = binary_dilation(in_img.copy(), footprint=close_fp)
        else:
            in_img = binary_dilation(in_img.copy())
    in_img = hole_filling(in_img.copy(), hole_min=Min_Hole_Width, hole_max=Max_Hole_Width, fill_2d=True)
    if Method == "Disk":
        if Size > 0:
            for z in range(len(in_img)):
                in_img[z] = binary_erosion(in_img.copy()[z], footprint=close_fp)
        else:
            for z in range(len(in_img)):
                in_img[z] = binary_erosion(in_img.copy()[z])
    elif Method == "Ball":
        if Size > 0:
            in_img = binary_erosion(in_img.copy(), footprint=close_fp)
        else:
            in_img = binary_erosion(in_img.copy())
    return in_img