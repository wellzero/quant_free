import json
import re
import os
import subprocess

def xq_js_to_dict(market = "us"):
    """
    Convert a JavaScript dictionary from a file to a Python dictionary, removing '$' and the following letter from keys.

    Parameters:
    js_file_path (str): The path to the JavaScript file.

    Returns:
    dict: The converted Python dictionary.
    """
    js_file_path =  os.path.join(os.getenv("QUANT_FREE_ROOT"), 'quant_free/utils/js/xq_finance_dict.js')
    try:
        # Execute the JavaScript file using Node.js
        result = subprocess.run(['node', js_file_path], capture_output=True, text=True)

        # Check if there was any output
        if result.stdout:
            # Load the JSON output into a Python dictionary
            data_dict = json.loads(result.stdout)
            # data_dict = json.dumps(table, indent=2, ensure_ascii=False)
        else:
            print("No output from the JavaScript file.")
            print("Error:", result.stderr)

        data_dict = deduplicate_values(data_dict)
       
        data = data_dict[market]
        
        def process_data(data, prefix):
            result = {}
            for key, value in data.items():
                if key.startswith(prefix):
                    result.update(value)
            return {re.sub(r'\$[^$]*', '', k): v for k, v in result.items()}

        result = {}
        result["metrics"] = process_data(data, 'indicator')
        result["income"] = process_data(data, 'income')
        result["balance"] = process_data(data, 'balance')
        result["cash"] = process_data(data, 'cash')

        return result
    
    except Exception as e:
        print(f'Error: {e}')
        return None

def deduplicate_values(data_dict):
    """
    Deduplicate values in the dictionary for the same keys, keeping the first occurrence.

    Parameters:
    data_dict (dict): The input dictionary with potential duplicate keys.

    Returns:
    dict: The processed dictionary with deduplicated values.
    """

    deduped_dict = {}
    for key, value in data_dict.items():
        for key1, value1 in value.items():
            for key2, value2 in value1.items():
                if key2 not in deduped_dict:
                    deduped_dict[key2] = value2
                else:
                    data_dict[key][key1][key2] = deduped_dict[key2]
    return data_dict

if __name__ == "__main__":
    data = xq_js_to_dict(market = "us")
    print(len(data['cash']))
