

def backtest_store_result(path, parameters):
    result_folder = path /'result'/ \
                    f'{parameters["symbol"]}_{parameters["factor_name"]}_{parameters["model"]}'
    print("result folder: ", result_folder)
    # Create result folder if it doesn't exist
    result_folder.mkdir(parents=True, exist_ok=True)
    
    # Move logs folder
    logs_folder = path /'result'/'logs'
    if logs_folder.exists():
        for log_file in logs_folder.iterdir():
            log_file.rename(result_folder / log_file.name)
        logs_folder.rmdir()
