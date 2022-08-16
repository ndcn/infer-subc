"""
infer_subc base module.

This is the principal module of the infer_subc project.
here you put your main classes and objects.
"""
import numpy as np

# example constant variable
NAME = "infer_subc"


# these are the fuinctions that need to be set by each notebook...
# notebook workflow will produce the in_params dictionary nescessary 
# so all the images can be pushed through these functions (procedures)

def infer_NUCLEI(struct_img, out_path, in_params):
    pass

def infer_SOMA(struct_img,  nuclei_labels, out_path, in_params):
    pass

def infer_CYTOSOL(struct_img, nuclei_labels, soma_labels, out_path, in_params):
    pass

def infer_LYSOSOMES(struct_img, out_path, cyto_labels, in_params):
    pass

def infer_MITOCHONDRIA(struct_img, out_path, cyto_labels, in_params):
    pass

def infer_GOLGI(struct_img, out_path, cyto_labels, in_params):
    pass

def infer_PEROXISOMES(struct_img, out_path, cyto_labels, in_params):
    pass

def infer_ENDOPLASMIC_RETICULUM(struct_img, out_path, cyto_labels, in_params):
    pass

def infer_LIPID_DROPLET(struct_img, out_path, cyto_labels, in_params):
    pass


#### Median FIltering for 2D
# We need to define a wrapper for `median_filter` which steps through each Z-slice independently.  (Note: since we will use this 
# pattern repeatedly we may want to make a generic wrapper for our filtering/de-noising). Lets call it `median_filter_slice_by_slice` 
# and copy the way the `aicssegmentation` package handles smoothing.
# TODO: typehints.... what is my "image" primitive?

def median_filter_slice_by_slice(struct_img, size):
    """
    wrapper for applying 2D median filter slice by slice on a 3D image
    """
    from scipy.ndimage import median_filter
    structure_img_denoise = np.zeros_like(struct_img)
    for zz in range(struct_img.shape[0]):
        structure_img_denoise[zz, :, :] = median_filter(struct_img[zz, :, :], size=size) 
        
    return structure_img_denoise


def simple_intensity_normalization(struct_img, max_value=None):
    """Normalize the intensity of input image so that the value range is from 0 to 1.

    Parameters:
    ------------
    img: np.ndarray
        a 3d image
    max_value: float
        
    """
    if max_value is not None:
        struct_img[struct_img > max_value] = max_value

    strech_min = struct_img.min()
    strech_max = struct_img.max()

    struct_img = (struct_img - strech_min + 1e-8) / (strech_max - strech_min + 1e-8)

    return struct_img


def read_input_image(image_name):
    # from aicsimageio import AICSImage
    # from napari_aicsimageio.core import _get_full_image_data, _get_meta
    # img_in = AICSImage(czi_image_name)
    # data_out = _get_full_image_data(img_in, in_memory=True)
    # meta_out = _get_meta(data, img_in)
    # meta_out['AICSImage'] = img_in

    # prefer this wrapper because it returns numpy arrays
    # or more simply with napari_aicsimagie io
    from napari_aicsimageio.core import  reader_function
    data_out, meta_out, layer_type = reader_function(image_name)[0]
    return (data_out,meta_out)



def log_transform(image):
    """Renormalize image intensities to log space
    
    Returns a tuple of transformed image and a dictionary to be passed into
    inverse_log_transform. The minimum and maximum from the dictionary
    can be applied to an image by the inverse_log_transform to 
    convert it back to its former intensity values.
    """
    orig_min, orig_max = scipy.ndimage.extrema(image)[:2]
    #
    # We add 1/2 bit noise to an 8 bit image to give the log a bottom
    #
    limage = image.copy()
    noise_min = orig_min + (orig_max - orig_min) / 256.0 + np.finfo(image.dtype).eps
    limage[limage < noise_min] = noise_min
    d = {"noise_min": noise_min}
    limage = np.log(limage)
    log_min, log_max = scipy.ndimage.extrema(limage)[:2]
    d["log_min"] = log_min
    d["log_max"] = log_max
    return stretch(limage), d


def inverse_log_transform(image, d):
    """Convert the values in image back to the scale prior to log_transform
    
    image - an image or value or values similarly scaled to image
    d - object returned by log_transform
    """
    return np.exp(unstretch(image, d["log_min"], d["log_max"]))


def stretch(image, mask=None):
    """Normalize an image to make the minimum zero and maximum one
    image - pixel data to be normalized
    mask  - optional mask of relevant pixels. None = don't mask
    returns the stretched image
    """
    image = np.array(image, float)
    if np.product(image.shape) == 0:
        return image
    if mask is None:
        minval = np.min(image)
        maxval = np.max(image)
        if minval == maxval:
            if minval < 0:
                return np.zeros_like(image)
            elif minval > 1:
                return np.ones_like(image)
            return image
        else:
            return (image - minval) / (maxval - minval)
    else:
        significant_pixels = image[mask]
        if significant_pixels.size == 0:
            return image
        minval = np.min(significant_pixels)
        maxval = np.max(significant_pixels)
        if minval == maxval:
            transformed_image = minval
        else:
            transformed_image = (significant_pixels - minval) / (maxval - minval)
        result = image.copy()
        image[mask] = transformed_image
        return image


def unstretch(image, minval, maxval):
    """Perform the inverse of stretch, given a stretched image
    image - an image stretched by stretch or similarly scaled value or values
    minval - minimum of previously stretched image
    maxval - maximum of previously stretched image
    """
    return image * (maxval - minval) + minval


def threshold_li_log( image_in ):
    """
    thin wrapper to log-scale and inverse log image for threshold finding using li minimum cross-entropy
    """
    from skimage.filters import threshold_li

    image, d = log_transform(image_in.copy())
    threshold =  threshold_li(image)
    threshold = inverse_log_transform(threshold, d)
    return threshold

    image_out = log_transform(image_in.copy())
    li_thresholded = structure_img_smooth >threshold_li_log( structure_img_smooth )

def threshold_otsu_log( image_in ):
    """
    thin wrapper to log-scale and inverse log image for threshold finding using otsu
    """
    from skimage.filters import threshold_otsu

    image, d = log_transform(image_in.copy())
    threshold =  threshold_otsu(image)
    threshold = inverse_log_transform(threshold, d)
    return threshold

def export_ome_tiff(data_in, meta_in, img_name, out_path, curr_chan=0) ->  str:
    #  data_in: types.ArrayLike,
    #  meta_in: dict,
    # img_name: types.PathLike,
    # out_path: types.PathLike,
    # curr_chan: int
    # assumes a single image
    from aicsimageio.writers import OmeTiffWriter

    out_name = out_path + img_name + ".ome.tiff"
   
    image_names = [img_name]
    print(image_names)
    #chan_names = meta_in['metadata']['aicsimage'].channel_names

    physical_pixel_sizes = [ meta_in['metadata']['aicsimage'].physical_pixel_sizes ]

    dimension_order = ["CZYX"]
    channel_names= [ meta_in['metadata']['aicsimage'].channel_names[curr_chan] ]
    if len(data_in.shape) == 3: #single channel zstack
        data_in=data_in[np.newaxis,:,:,:]
    
    if data_in.dtype == 'bool':
        data_in = data_in.astype(np.uint8)
        data_in[ data_in > 0 ] = 255

    out_ome = OmeTiffWriter.build_ome(
                    [ data_in.shape],
                    [data_in.dtype],
                    channel_names=channel_names,  # type: ignore
                    image_name=image_names,
                    physical_pixel_sizes=physical_pixel_sizes,
                    dimension_order=dimension_order,
                )


    OmeTiffWriter.save( data_in,
                                        out_name,
                                        dim_order=dimension_order,
                                        channel_names = channel_names,
                                        image_names = image_names,
                                        physical_pixel_sizes = physical_pixel_sizes,
                                        ome_xml=out_ome,
                        )
    return out_name


### UTILS
from collections import defaultdict
def etree_to_dict(t):
    d = {t.tag: {} if t.attrib else None}
    children = list(t)
    if children:
        dd = defaultdict(list)
        for dc in map(etree_to_dict, children):
            for k, v in dc.items():
                dd[k].append(v)
        d = {t.tag: {k: v[0] if len(v) == 1 else v
                     for k, v in dd.items()}}
    if t.attrib:
        d[t.tag].update(('@' + k, v)
                        for k, v in t.attrib.items())
    if t.text:
        text = t.text.strip()
        if children or t.attrib:
            if text:
                d[t.tag]['#text'] = text
        else:
            d[t.tag] = texts
    return d



def get_raw_meta_data(meta_dict):
    from platform import system
    curr_platform = system()

    if curr_platform=='Linux':
        raw_meta_data = meta_dict['metadata']['raw_image_metadata'].dict()
        ome_types = meta_dict['metadata']['ome_types']
    elif curr_platform=='Darwin':
        raw_meta_data = meta_dict['metadata']['raw_image_metadata']
        ome_types = []
    else:
        raw_meta_data = meta_dict['metadata']['raw_image_metadata']
        ome_types = []
        print(f"warning: platform = '{curr_platform}' is untested")
    return (raw_meta_data,ome_types)






