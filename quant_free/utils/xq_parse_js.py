import json
import re
import os

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
        # Read the JavaScript file
        with open(js_file_path, 'r', encoding='utf-8') as file:
            js_content = file.read()

        # Remove the JavaScript variable assignment to isolate the object
        js_content_1 = re.sub(r'\$[^:]*:', ':', js_content)
        js_content_2 = re.sub(r'subtitle\d+: "[^"]*",\n', '', js_content_1)

        js_object_str = re.sub(r'^[^=]*=\s*', '', js_content_2, count=1).strip().rstrip(';')

        # Replace single quotes with double quotes for JSON compatibility
        json_str = js_object_str.replace("'", '"')

        # Fix other JavaScript to JSON incompatibilities, if any
        json_str = re.sub(r'(\w+):', r'"\1":', json_str)  # Add double quotes around keys

        # Parse the JSON string to a Python dictionary
        data_dict = json.loads(json_str)

        # Remove '$' and the following letter from keys
        def clean_key(key):
            return re.sub(r'\$.', '', key)
        
        cleaned_dict = {clean_key(k): v for k, v in data_dict.items()}
        
        data = cleaned_dict[market]
        
        def process_data(data, prefix):
            result = {}
            for key, value in data.items():
                if key.startswith(prefix):
                    result.update(value)
            return {re.sub(r'\$[^$]*', '$', k): v for k, v in result.items()}

        result = {}
        result["metrics"] = process_data(data, 'indicator')
        result["income"] = process_data(data, 'income')
        result["balance"] = process_data(data, 'balance')
        result["cash"] = process_data(data, 'cash')

        return result
    
    except Exception as e:
        print(f'Error: {e}')
        return None

if __name__ == "__main__":
    data = xq_js_to_dict(market = "us")
    print(len(data))
