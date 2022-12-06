from aicssegmentation.workflow import (
    WorkflowEngine,
    SegmenterFunction,
    PrebuiltWorkflowDefinition,
    WorkflowDefinition,
    Workflow,
)
from aicssegmentation.workflow.workflow_config import WorkflowConfig, ConfigurationException

import json
from pathlib import Path

import infer_subc_2d

from aicssegmentation.workflow import WorkflowStep, WorkflowStepCategory

import importlib
import numpy as np

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Any
from aicssegmentation.workflow.segmenter_function import SegmenterFunction

# TODO:  rather than subclassing we should try an ABC composition?

"""
some subclassing of aicssegmentation.workflow  for use in this repo

"""

# Directories
class InferSubC2dDirectories:
    """
    Provides safe paths to common infer-subc-2D module directories
    """

    _module_base_dir = Path(infer_subc_2d.__file__).parent

    @classmethod
    def get_assets_dir(cls) -> Path:
        """
        Path to the assets directory
        """
        return cls._module_base_dir / "assets"

    @classmethod
    def get_structure_config_dir(cls) -> Path:
        """
        Path to the structure json config directory
        """
        return cls._module_base_dir / "organelles_config"


# WorkflowStep
class InferSubC2dWorkflowStepCategory(Enum):  # cannot extend WorkflowStepCategory which is Enum
    EXTRACTION = "extraction"
    PRE_PROCESSING = "preprocessing"
    CORE = "core"
    POST_PROCESSING = "postprocessing"
    POST_POST_PROCESSING = "postpostprocessing"

    @staticmethod
    def from_str(value: str):
        if value is not None:
            value = value.lower()
        if value == InferSubC2dWorkflowStepCategory.EXTRACTION.value:
            return InferSubC2dWorkflowStepCategory.EXTRACTION
        if value == InferSubC2dWorkflowStepCategory.PRE_PROCESSING.value:
            return InferSubC2dWorkflowStepCategory.PRE_PROCESSING
        if value == InferSubC2dWorkflowStepCategory.CORE.value:
            return InferSubC2dWorkflowStepCategory.CORE
        if value == InferSubC2dWorkflowStepCategory.POST_PROCESSING.value:
            return InferSubC2dWorkflowStepCategory.POST_PROCESSING
        if value == InferSubC2dWorkflowStepCategory.POST_POST_PROCESSING.value:
            return InferSubC2dWorkflowStepCategory.POST_POST_PROCESSING
        raise NotImplementedError()


@dataclass
class InferSubC2dWorkflowStep(WorkflowStep):
    """
    Represents a single step in an aicssegmentation Workflow
    """

    category: InferSubC2dWorkflowStepCategory
    function: SegmenterFunction
    step_number: int
    parent: List[int]
    parameter_values: Dict[str, List] = None


# WorkflowDefinition
@dataclass
class InferSubC2dWorkflowDefinition(WorkflowDefinition):
    """
    Definition of a custom aics-segmentation Workflow loaded from file.

    This class only defines the workflow (i.e. the workflow characteristics and steps)
    and is used either for building an executable Workflow object
    or to access information about the Workflow without needing to execute it
    """

    name: str
    steps: List[InferSubC2dWorkflowStep]

    def __init__(self, name: str, steps: List[InferSubC2dWorkflowStep]):
        self.name = name
        self.steps = steps
        self.from_file = True


@dataclass
class InferSubC2dPrebuiltWorkflowDefinition(InferSubC2dWorkflowDefinition):
    """
    Definition of a pre-built(default) aics-segmentation Workflow from our assets.

    This class only defines the workflow (i.e. the workflow characteristics and steps)
    and is used either for building an executable Workflow object
    or to access information about the Workflow without needing to execute it
    """

    def __init__(self, name: str, steps: List[InferSubC2dWorkflowStep]):
        WorkflowDefinition.__init__(self, name=name, steps=steps)


# WorkflowConfig (workflow.workflow_config)
class InferSubC2dWorkflowConfig(WorkflowConfig):
    """
    infer-subc-2D Provides access to structure workflow configuration
    """

    def __init__(self):
        self._all_functions = None
        self._available_workflow_names = None

    def get_available_workflows(self) -> List[str]:
        """
        Get the list of all workflows available through configuration
        """
        if self._available_workflow_names is None:
            json_list = sorted(InferSubC2dDirectories.get_structure_config_dir().glob("conf_*.json"))
            self._available_workflow_names = [p.stem[5:] for p in json_list]

        return self._available_workflow_names

    def get_all_functions(self) -> List[SegmenterFunction]:
        """
        Get the list of all available Functions from configuration
        """
        if self._all_functions is None:
            path = InferSubC2dDirectories.get_structure_config_dir() / "all_functions.json"

            try:
                with open(path) as file:
                    obj = json.load(file)
                    self._all_functions = self._all_functions_decoder(obj)
            except Exception as ex:
                raise ConfigurationException(f"Error reading json configuration from {path}") from ex

        return self._all_functions

    def get_workflow_definition(self, workflow_name: str) -> InferSubC2dPrebuiltWorkflowDefinition:
        """
        Get a WorkflowDefinition for the given workflow from the corresponding
        prebuilt json structure config
        """
        if workflow_name is None or len(workflow_name.strip()) == 0:
            raise ValueError("workflow_name cannot be empty")

        if workflow_name not in self.get_available_workflows():
            raise ValueError(f"No workflow configuration available for {workflow_name}")

        path = InferSubC2dDirectories.get_structure_config_dir() / f"conf_{workflow_name}.json"

        return self.get_workflow_definition_from_config_file(path, workflow_name, prebuilt=True)

    def get_workflow_definition_from_config_file(
        self, file_path: Path, workflow_name: str = None, prebuilt: bool = False
    ) -> InferSubC2dWorkflowDefinition:
        """
        Get a WorkflowDefinition based off the given json configuration file
        """
        if file_path.suffix.lower() != ".json":
            raise ValueError("Workflow configuration file must be a json file with .json file extension.")

        with open(file_path) as file:
            try:
                obj = json.load(file)
                return self._workflow_decoder(obj, workflow_name or file_path.name, prebuilt)
            except Exception as ex:
                raise ConfigurationException(f"Error reading json configuration from {file_path}") from ex

    def _workflow_decoder(self, obj: Dict, workflow_name: str, prebuilt: bool = False) -> InferSubC2dWorkflowDefinition:
        """
        Decode Workflow config (conf_{workflow_name}.json)
        """
        functions = self.get_all_functions()
        steps: List[InferSubC2dWorkflowStep] = list()

        for step_k, step_v in obj.items():
            step_number = int(step_k)
            function_id = step_v["function"]
            function = next(filter(lambda f: f.name == function_id, functions), None)

            if function is None:
                raise ConfigurationException(
                    f"Could not find a Segmenter function matching the function identifier <{function_id}>."
                )

            if isinstance(step_v["parent"], list):
                parent = step_v["parent"]
            else:
                parent = [step_v["parent"]]
            step = InferSubC2dWorkflowStep(
                category=InferSubC2dWorkflowStepCategory.from_str(step_v["category"]),
                function=function,
                step_number=step_number,
                parent=parent,
            )

            if step_v.get("parameter_values") is not None and len(step_v["parameter_values"]) > 0:
                param_defaults = dict()

                for param_k, param_v in step_v["parameter_values"].items():
                    param_name = param_k
                    param_defaults[param_name] = param_v

                step.parameter_values = param_defaults

            steps.append(step)

        steps.sort(key=lambda s: s.step_number)

        if prebuilt:
            return InferSubC2dPrebuiltWorkflowDefinition(workflow_name, steps)
        else:
            return InferSubC2dWorkflowDefinition(workflow_name, steps)

    def _workflow_encoder(self, workflow_definition: InferSubC2dWorkflowDefinition) -> Dict:
        """
        Encode a WorkflowDefinition to a json dictionary
        """

        # TODO add header / version ?
        result = dict()
        for step in workflow_definition.steps:
            step_number = str(step.step_number)
            parent = step.parent[0] if len(step.parent) == 1 else step.parent

            step_dict = {
                step_number: {"function": step.function.name, "category": step.category.value, "parent": parent}
            }
            if step.parameter_values is not None:
                step_dict[step_number].update({"parameter_values": step.parameter_values})

            result.update(step_dict)

        return result


class InferSubC2dWorkflowEngine(WorkflowEngine):
    """
    infer-subc-2D workflow engine
    Use this class to access and execute aicssegmentation structure workflows
    """

    def __init__(self, workflow_config: WorkflowConfig = None):
        self._workflow_config = workflow_config or InferSubC2dWorkflowConfig()
        self._workflow_definitions = self._load_workflow_definitions()

    def _load_workflow_definitions(self) -> List[InferSubC2dWorkflowDefinition]:
        definitions = list()
        available_workflows = self._workflow_config.get_available_workflows()
        for name in available_workflows:
            definitions.append(self._workflow_config.get_workflow_definition(name))
        return definitions


# Workflow base
class InferSubC2dWorkflow(Workflow):
    """
    Represents an executable aics-segmentation workflow
    This class provides the functionality to run a workflow using an image input
    according to the steps defined in its WorkflowDefinition.
    """

    def __init__(self, workflow_definition: InferSubC2dWorkflowDefinition, input_image: np.ndarray):
        if workflow_definition is None:
            raise ArgumentNullError("workflow_definition")
        if input_image is None:
            raise ArgumentNullError("input_image")
        self._definition: InferSubC2dWorkflowDefinition = workflow_definition
        self._starting_image: np.ndarray = input_image
        self._next_step: int = 0  # Next step to execute
        self._results: List = list()  # Store most recent step results

    @property
    def workflow_definition(self) -> InferSubC2dWorkflowDefinition:
        return self._definition

    def reset(self):
        """
        Reset the workflow so it can be run again
        """
        self._next_step = 0
        self._results = list()

    def get_next_step(self) -> InferSubC2dWorkflowDefinition:
        """
        Get the next step to be performed

        Params:
            none

        Returns:
            (WorkflowStep): next WorkflowStep object to perform on image
            None if all steps have already been executed
        """
        if self._next_step >= len(self._definition.steps):
            return None
        return self._definition.steps[self._next_step]
