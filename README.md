# quant_free

This project used to spider US stock daily trade data and financial data freely, and do some statistcal research for history trade and financial data. this project only used to research US equity.

## install
### requirement installation
pip install -r requirements.txt
### install chrome for efinance downlaod
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo apt install ./google-chrome-stable_current_amd64.deb   
### spider test

```python
>>> from selenium import webdriver
>>> from selenium.webdriver.chrome.service import Service
>>> from webdriver_manager.chrome import ChromeDriverManager
>>> from selenium.webdriver.chrome.options import Options
>>> from selenium.webdriver.common.by import By
>>> from selenium.webdriver.common.keys import Keys
 
 
>>> from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
>>> from selenium.webdriver.support.ui import WebDriverWait
>>> from selenium.webdriver.support import expected_conditions as EC
 
>>> options = Options()
>>> options.add_argument('--headless')
>>> options.add_argument('--no-sandbox')
>>> options.add_argument('--disable-dev-shm-usage')
>>> driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
```

## configuration
configure folder stored download data
```bash
vim conf.json
{"data_dir": "/your_dir/quant_free/data"}
```

## download data
download data to data_dir.
```bash
cd test
python test_us_equity_xq_finance_download.py
```

## some jupyter research
there is some statistical analysis in research folder for US equity.

## QuantRL
Cloning the Repository with Submodules
```bash
 git submodule update --init --recursive
```
Updating Submodules
```bash
 git submodule update --remote
```






