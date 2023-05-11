import numpy as np

from tifffile import imwrite #, tiffcomment, imread


from datetime import datetime
from typing import List, Union
# from aicsimageio import AICSImage
# from aicsimageio.writers import OmeTiffWriter
from pathlib import Path


# from aicssegmentation.util.filesystem import FileSystemUtilities
# from aicssegmentation.exceptions import ArgumentNullError
from infer_subc.utils.filesystem import FileSystemUtilities
from infer_subc.exceptions import ArgumentNullError
from infer_subc.workflow.workflow import Workflow
from infer_subc.workflow.workflow_definition import WorkflowDefinition

# from infer_subc.core.file_io import reader_function
from infer_subc.core.file_io import reader_function

PathLike = Union[str, Path]
SUPPORTED_FILE_EXTENSIONS = ["tiff", "tif", "czi"]


# TODO: fix channel index.
class BatchWorkflow:
    """
    Represents a batch of workflows to process.
    This class provides the functionality to run batches of workflows using multiple image inputs from a input directory
    according to the steps defined in its WorkflowDefinition.
    """

    def __init__(
        self,
        workflow_definitions: List[WorkflowDefinition],
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        segmentation_names: List[str],  # JAH: add segmentation name for export
        channel_index: int = -1,  # JAH: change so all negative indices return ALL the channels/zslices
    ):
        if workflow_definitions is None:
            raise ArgumentNullError("workflow_definitions")
        if segmentation_names is None:
            raise ArgumentNullError("segmentation_names")
        if input_dir is None:
            raise ArgumentNullError("input_dir")
        if output_dir is None:
            raise ArgumentNullError("output_dir")

        self._workflow_definitions = workflow_definitions
        self._input_dir = Path(input_dir)
        self._segmentation_names = segmentation_names

        if not self._input_dir.exists():
            raise ValueError("The input directory does not exist")

        self._output_dir = Path(output_dir)
        self._channel_index = channel_index
        self._processed_files: int = 0
        self._failed_files: int = 0
        self._log_path: Path = self._output_dir / f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        # Create the output directory at output_dir if it does not exist already
        if not self._output_dir.exists():
            FileSystemUtilities.create_directory(self._output_dir)

        self._input_files = self._get_input_files(self._input_dir, SUPPORTED_FILE_EXTENSIONS)
        self._execute_generator = self._execute_generator_func()

    @property
    def total_files(self) -> int:
        return len(self._input_files) * len(self._workflow_definitions)

    @property
    def processed_files(self) -> int:
        return self._processed_files

    @property
    def failed_files(self) -> int:
        return self._failed_files

    @property
    def input_dir(self) -> Path:
        return self._input_dir

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    @property
    def segmentation_names(self) -> List[str]:
        return self._segmentation_names

    def is_done(self) -> bool:
        """
        Indicates whether all files / steps have been executed

        Use this to know when the batch workflow is complete if manually executing the workflow
        with execute_next()

        Returns
            (bool): True if all files/steps have been executed, False if not
        """
        return self._processed_files == self.total_files

    def execute_all(self):
        if self.is_done():
            print("No files left to process")
            return

        print("Starting batch workflow...")
        print(f"Found {self.total_files} files X workflows to process.")

        while not self.is_done():
            self.execute_next()

        self.write_log_file_summary()

        print(f"Batch workflow complete. Check {self._log_path} for output log and summary.")

    def execute_next(self):
        if self.is_done():
            print("No files left to process")
            return

        next(self._execute_generator)

    def _execute_generator_func(self):
        for f in self._input_files:
            msg = f"\nprocessing::: {f.name} :"
            self._write_to_log_file(msg)
            for wf, seg_nm in zip(self._workflow_definitions, self._segmentation_names):
                try:
                    # read and format image in the way we expect
                    # read_image = AICSImage(f)
                    # image_from_path = self._format_image_to_3d(read_image)
                    image_from_path = self._format_image_to_3d(f)

                    # Run workflow on image
                    msg = f": {seg_nm} :"
                    self._write_to_log_file(msg)

                    workflow = Workflow(wf, image_from_path)
                    while not workflow.is_done():
                        workflow.execute_next()
                        result = workflow.get_most_recent_result()
                        # print(f"current result shape + type: {result.shape}, {result.dtype}")
                        yield

                    # Save output
                    # output_path = self._output_dir / f"{f.stem}.segmentation.tiff"
                    output_path = self._output_dir / f"{f.stem}-{seg_nm}.tiff"

                    result = workflow.get_most_recent_result()
                    result = self._format_output(result)

                    ret = imwrite(
                        output_path,
                        result,
                        dtype=result.dtype,
                        # metadata={
                        #     "axes": dimension_order,
                        #     # "physical_pixel_sizes": physical_pixel_sizes,
                        #     # "channel_names": channel_names,
                        # },
                    )
                    # if len(result.shape) == 3:
                    #     # TODO:  replace with. tifffile writer ...
                    #     OmeTiffWriter.save(data=self._format_output(result), uri=output_path, dim_order="ZYX")
                    # else:
                    #     OmeTiffWriter.save(data=self._format_output(result), uri=output_path, dim_order="CZYX")

                    msg = f"SUCCESS: {f}:{seg_nm}. >>>> {output_path.name}"
                    print(msg)
                    self._write_to_log_file(msg)

                except Exception as ex:
                    self._failed_files += 1
                    msg = f"FAILED: {f}:{seg_nm}, ERROR: {ex}"
                    print(msg)
                    self._write_to_log_file(msg)
                finally:
                    self._processed_files += 1

                yield

    def write_log_file_summary(self):
        """
        Write a log file to the output folder.
        """
        if self._processed_files == 0:
            report = (
                f"There were no files to process in the input directory to process \n "
                f"Using the Workflow: {self._workflow_definition.name}"
            )
        else:
            files_processed = self._processed_files - self._failed_files
            wfs = ", ".join([wfd.name for wfd in self._workflow_definitions])
            report = (
                f"{files_processed}/{self._processed_files} files were successfully processed \n "
                f"Using the Workflows: {wfs}"
            )
        self._write_to_log_file(report)

    # def _format_image_to_3d(self, image: AICSImage) -> np.ndarray:
    def _format_image_to_3d(self, image_path: PathLike) -> np.ndarray:
        """
        Format images in the way that aics-segmention expects for most workflows (3d, zyx)

        Params:
            image_path (Path or str): image to format

        Returns
            np.ndarray: segment-able image for aics-segmentation
        """
        # TODO: make this reading more performant
        # if len(image.scenes) > 1:
        #     raise ValueError("Multi-Scene images are unsupported")

        # if image.dims.T > 1:
        #     raise ValueError("Timelapse images are unsupported.")

        # if image.dims.C > 1:
        #     return image.get_image_data("ZYX", C=self._channel_index)

        # return image.get_image_data("ZYX")
        # JAH: refactdor to use reader_function
        data, meta, layer_type = reader_function(image_path)[0]

        if isinstance(image_path, str):
            image_path = Path(image_path)
        name = image_path.stem

        channel_names = meta.pop("name")  # list of names for each layer
        meta["channel_names"] = channel_names
        meta["file_name"] = name
        layer_attributes = {"name": name, "metadata": meta}

        # TODO:  dump metadata dictionary. json or pickle?

        # return [(data, layer_attributes, layer_type)]  # (data,meta) is fine since 'image' is default
        # JAH: refactor to make channel_index be a zslice

        print(f"loaded {name} size: {data.shape}")
        if self._channel_index < 0:
            return data
        else:
            return data[:, self._channel_index, :, :]

    def _format_output(self, image: np.ndarray):
        """
        Format segmented images to uint8 to save via AICSImage

        Params:
            image (np.ndarray): segmented image

        Returns
            np.ndarray: image converted to uint8 for saving if boolean... otherwise
        """
        if image.dtype == "bool" or image.dtype == np.uint8:
            image = image.astype(np.uint16)
            image[image > 0] = 1
            msg = f"converted boolean to {image.dtype}. "
            self._write_to_log_file(msg)
        else:
            image = image.astype(np.uint16)
            msg = f" enforced  {image.dtype}"
            print(msg)
            self._write_to_log_file(msg)

        return image

    def _write_to_log_file(self, text: str):
        with open(self._log_path, "a") as writer:
            writer.write(f"{text}\n")

    def _get_input_files(self, input_dir: Path, extensions: List[str]) -> List[Path]:
        input_files = list()
        for ext in extensions:
            input_files.extend(input_dir.glob(f"*.{ext}"))
        return input_files
