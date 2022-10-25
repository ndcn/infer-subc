from aicssegmentation.workflow import WorkflowEngine, SegmenterFunction, PrebuiltWorkflowDefinition
from aicssegmentation.workflow.workflow_config import WorkflowConfig, ConfigurationException
from typing import List

import json
from pathlib import Path

import infer_subc_2d

"""
some subclassing of aicssegmentation.workflow  for use in this repo

"""


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

    def get_workflow_definition(self, workflow_name: str) -> PrebuiltWorkflowDefinition:
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


class InferSubC2dWorkflowEngine(WorkflowEngine):
    """
    infer-subc-2D workflow engine
    Use this class to access and execute aicssegmentation structure workflows
    """

    def __init__(self, workflow_config: WorkflowConfig = None):
        self._workflow_config = workflow_config or InferSubC2dWorkflowConfig()
        self._workflow_definitions = self._load_workflow_definitions()
