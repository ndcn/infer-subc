import numpy as np

# from aicsimageio import imread
from typing import Dict, Tuple, Any, Union, List
from dataclasses import dataclass
# from infer_subc.utils.lazy import lazy_property  # infer_subc.utils.lazy
from infer_subc.utils.directories import Directories  # infer_subc.utils.directories
from infer_subc.workflow.workflow_step import WorkflowStep


# TODO: create an WorkflowImage class
# TODO: make a separate python file for this


@dataclass
class SegmentationWrap:
    """
    Simple dataclass wrapper for segmentations of organelles  + masks
    TODO: make a nice reppr
    """

    name: str
    image: np.ndarray
    meta: Dict[str, Any]
    raw_meta: Tuple[Dict[str, Any], Union[Dict[str, Any], List]]
    channel_names: List[str]
    channels: List[int]
    segmentations: List[np.ndarray]
    masks: List[np.ndarray]
    mask_names: List[int]

    def __init__(self, name: str, image: np.ndarray, meta: Dict[str, Any]):
        self.name = name
        self.image = image
        self.meta = meta
        # self.raw_meta = get_raw_meta_data(meta)

    def add_mask(self, name: str, mask: np.ndarray):
        self.mask_names.append(name)
        self.masks.append(mask)

    def add_segmentation(self, name: str, segmentation: np.ndarray, channel: int):
        self.channel_names.append(name)
        self.channels.append(channel)
        self.segmentations.append(segmentation)


@dataclass
class WorkflowDefinition:
    """
    Definition of a custom aics-segmentation Workflow loaded from file.

    This class only defines the workflow (i.e. the workflow characteristics and steps)
    and is used either for building an executable Workflow object
    or to access information about the Workflow without needing to execute it
    """

    name: str
    steps: List[WorkflowStep]
    prebuilt: bool

    def __init__(self, name: str, steps: List[WorkflowStep], prebuilt: bool = True):
        self.name = name
        self.steps = steps
        self.prebuilt = prebuilt
        self.from_file = True


# depricate Prebuilt flavor.  we will not be calling the pictures at this point...
# @dataclass
# class PrebuiltWorkflowDefinition(WorkflowDefinition):
#     """
#     Definition of a pre-built(default) aics-segmentation Workflow from our assets.

#     This class only defines the workflow (i.e. the workflow characteristics and steps)
#     and is used either for building an executable Workflow object
#     or to access information about the Workflow without needing to execute it
#     """

#     def __init__(self, name: str, steps: List[WorkflowStep]):
#         WorkflowDefinition.__init__(self, name=name, steps=steps)

#     @lazy_property
#     def thumbnail_pre(self) -> np.ndarray:
#         """
#         The Pre-segmentation thumbnail related to this workflow, as a numpy array
#         """
#         return np.squeeze(imread(Directories.get_assets_dir() / f"thumbnails/{self.name.lower()}_pre.png"))

#     @lazy_property
#     def thumbnail_post(self) -> np.ndarray:
#         """
#         The Post-segmentation thumbnail related to this workflow, as a numpy array
#         """
#         return np.squeeze(imread(Directories.get_assets_dir() / f"thumbnails/{self.name.lower()}_post.png"))

#     @lazy_property
#     def diagram_image(self) -> np.ndarray:
#         """
#         Diagram / flow chart image for this workflow, as a numpy array
#         """
#         return np.squeeze(imread(Directories.get_assets_dir() / f"diagrams/{self.name.lower()}.png"))
