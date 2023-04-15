# from aicssegmentation

import infer_subc

from pathlib import Path


class Directories:
    """
    Provides safe paths to common infer-subc module directories
    """

    _module_base_dir = Path(infer_subc.__file__).parent

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
