# quant_free

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


