import json
import os
import numpy as np


def export_to_json(name:str, result_list:list, print_out=True):
    """
    Exports each result dictionary to its own folder in "results"

    Arguments:
        out_name: the (code)name of the file generating the result_list (see examples in "verify" files or __main__ below)
        result_list: metric list returned by "verify" files
    """
    # Target directory: artifacts/results
    art = "./artifacts/results"
    out_dir = os.path.join(art, name)

    # Check if the directory already exists
    if not os.path.exists(out_dir):
        # Create the directory
        os.makedirs(out_dir)
        print(f"Directory {out_dir} created")

    # Get the statistics of the results
    avg = np.mean(result_list)
    std = np.std(result_list)

    # Print out the results if desired
    if print_out == True:
        print("metric_list:", result_list)
        print("mean:", avg)
        print("std:", std)

    # Creating the dictionary object
    out_dict = {"name": name, "metric_list": result_list,
                "mean": avg, "std": std}
    
    # Formatting the filename
    file_out = "output_" + str(len(os.listdir(out_dir))) + ".json"
    final_out_name = os.path.join(out_dir, file_out)

    # Exporting to JSON file
    with open(final_out_name, "w") as outfile:
        json.dump(out_dict, outfile, indent=2)

    print(f"Exported '{name}' results to '{file_out}'.")


# For testing
if __name__ == "__main__":

    # The below lines create a file "output_{num}.json" in "artifacts/results/verify_clip_pcbm_test"
    out_name = "verify_clip_pcbm_test"
    export_to_json(out_name, [0.9, 0.75, 0.8, 0.6])
