import numpy as np
from skimage.morphology import disk, ball, binary_closing, closing, binary_dilation, dilation, binary_erosion
from infer_subc.core.img import fill_and_filter_linear_size, get_interior_labels, get_max_label, inverse_log_transform, log_transform, masked_inverted_watershed, masked_object_thresh, min_max_intensity_normalization, select_channel_from_raw, threshold_otsu_log
from aicssegmentation.core.utils import hole_filling
from infer_subc.organelles.nuclei import infer_nuclei_fromlabel
from infer_subc.organelles.cellmask import non_linear_cellmask_transform

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
        raw_img[PM_Channel] = np.max(in_img[PM_Channel])-in_img[PM_Channel]
    for ch, weight in enumerate(weights):
        if weight > 0:
            out_img+=weight*min_max_intensity_normalization(raw_img[ch])
    return out_img

def masked_object_thresh_bind_pm(raw_img: np.ndarray,
                                 in_img: np.ndarray,
                                 Global_Method: str,
                                 Cutoff_Size: int,
                                 Local_Adjust: float,
                                 Bind_to_PM: bool,            
                                 PM_Channel: int,
                                 Thresh_Adj: float):
    threshed = masked_object_thresh(in_img, Global_Method, Cutoff_Size, Local_Adjust)

    if Bind_to_PM:
        pm_img = select_channel_from_raw(raw_img, PM_Channel)
        pm_image, d = log_transform(pm_img.copy())
        pm_thresh = threshold_otsu_log(pm_image) 
        invert_pm_obj = np.invert(pm_img > (inverse_log_transform(pm_thresh, d) * Thresh_Adj))
        mask = np.zeros_like(invert_pm_obj)
        mask[(invert_pm_obj == threshed) & 
                 (invert_pm_obj == True) & 
                 (threshed == True)] = 1
        out_img = closing(mask)
    else:
        out_img = threshed.copy()

    return out_img

def close_and_filter(in_img: np.ndarray, 
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
    elif Method == "Scharr":
        out_img = non_linear_cellmask_transform(in_img)
    elif Method == "None":
        out_img = in_img
    return out_img

def find_nuc(raw_img: np.ndarray, 
             in_img_A: np.ndarray,
             in_img_B: np.ndarray,
             Nuc_Channel: int,
             Median_Size: int,
             Gauss_Sigma: float,
             Thresh_Factor: float,
             Thresh_Min: float,
             Thresh_Max: float,
             Min_Hole_Width: int,
             Max_Hole_Width: int,
             Small_Obj_Width: int,
             Fill_Filter_Method: str,
             Search_Img: str
             ):
    nuc_img = infer_nuclei_fromlabel(in_img=raw_img, 
                                     nuc_ch=Nuc_Channel,
                                     median_size=Median_Size, 
                                     gauss_sigma=Gauss_Sigma,
                                     thresh_factor=Thresh_Factor,
                                     thresh_min=Thresh_Min,
                                     thresh_max=Thresh_Max,
                                     min_hole_width=Min_Hole_Width,
                                     max_hole_width=Max_Hole_Width,
                                     small_obj_width=Small_Obj_Width,
                                     fill_filter_method=Fill_Filter_Method)
    if Search_Img == "Img 5":
        in_img = in_img_A
    elif Search_Img == "Img 6":
        in_img = in_img_B
    keep_nuc = get_max_label((in_img), dilation(nuc_img))
    out_img = np.zeros_like(nuc_img)
    out_img[nuc_img == keep_nuc] = 1
    return out_img

def mix_nuc_and_fill(nuc: np.ndarray, 
            in_img: np.ndarray, 
            Method: str, 
            Size: int,
            Min_Hole_Width: int,
            Max_Hole_Width: int,
            Small_Obj_Width: int):
    single_nuc = nuc
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
    elif Method == "None":
        out_img += single_nuc.astype(bool)
    filled = fill_and_filter_linear_size(out_img,
                                         hole_min=Min_Hole_Width,
                                         hole_max=Max_Hole_Width,
                                         min_size=Small_Obj_Width)
    return filled

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
        mask[(binary_dilation(invert_pm_obj) > 0) & 
             (filled > 0)] = 1
        out_img = binary_closing(mask)
    else:
        out_img = filled.copy()
    return out_img

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

def double_watershed(nuc: np.ndarray,
                     raw_img_A: np.ndarray,
                     raw_img_B: np.ndarray,
                     thresh_img_A: np.ndarray,
                     thresh_img_B: np.ndarray,
                     Watershed_Method: str,
                     Min_Hole_Width: int,
                     Max_Hole_Width: int,
                     Method: str,
                     Size: int):
    
    choose_nuc = nuc

    cm_A = masked_inverted_watershed(raw_img_A, 
                                     choose_nuc, 
                                     thresh_img_A,
                                     method=Watershed_Method)
    cm_B = masked_inverted_watershed(raw_img_B, 
                                     choose_nuc, 
                                     thresh_img_B,
                                     method=Watershed_Method)
    
    cm_combo = cm_A.astype(bool) + cm_B.astype(bool)
    cm_out = close_and_fill(in_img=cm_combo,
                             Min_Hole_Width=Min_Hole_Width,
                             Max_Hole_Width=Max_Hole_Width,
                             Method=Method,
                             Size=Size)
    
    out_img = np.stack([nuc, cm_out], axis=0)

    return out_img

def invert_pm_watershed(raw_img: np.ndarray, nuc_labels: np.ndarray, PM_Channel: int, Method: str):
    pm_img = select_channel_from_raw(raw_img, PM_Channel)
    invert_pm_img = abs(np.max(pm_img) - pm_img)
    out_img = masked_inverted_watershed(img_in=min_max_intensity_normalization(invert_pm_img),
                                     markers=nuc_labels,
                                     mask=None,
                                     method=Method)
    return out_img

def choose_cell(composite_img: np.ndarray, nuc_labels: np.ndarray, watershed: np.ndarray):
    target_labels = get_interior_labels(nuc_labels)
    keep_label = get_max_label(composite_img, 
                           watershed,
                           target_labels=target_labels)
    cellmask_out = np.zeros_like(watershed)
    cellmask_out[watershed == keep_label] = 1
    return cellmask_out