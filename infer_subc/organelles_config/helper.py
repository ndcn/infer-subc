import json
from infer_subc.utils.directories import Directories


def add_function_spec_to_widget_json(
    function_name, function_dict, json_file_name="all_functions.json", overwrite=False
) -> int:
    """helper function to compose / update list of functions for Workflows"""
    # read all_functions.json into dict
    path = Directories.get_structure_config_dir() / json_file_name
    try:
        with open(path) as file:
            obj = json.load(file)
    except:  # Exception as ex:
        print(f"file {path} not found")
        return

    # add function entry
    if function_name in obj.keys():
        print(f"function {function_name} is already in {json_file_name}")
        if overwrite:
            print(f"overwriting  {function_name}")
        else:
            return 0

    obj[function_name] = function_dict  # write updated all_functions.json

    # re-write file
    with open(path, "w") as file:
        json.dump(obj, file, indent=4, sort_keys=False)

    return 1


def write_workflow_json(wf_name, wf_dict):
    """helper function to dump dictionary of workflows to "configuration" jsons"""
    # read all_functions.json into dict
    if not wf_name.startswith("conf"):
        wf_name = f"conf_{wf_name}"
    path = Directories.get_structure_config_dir() / f"{wf_name}.json"

    # re-write file
    with open(path, "w") as file:
        json.dump(wf_dict, file, indent=4, sort_keys=False)

    return path
