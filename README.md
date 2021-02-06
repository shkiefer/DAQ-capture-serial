# ohDAQ-capture-serial
Python web-app for capturing json-like data over serial.  It is based on [Plotly's Dash](https://plotly.com/dash/) framework for the web UI.
Data is assumed to be streaming over serial and each line is formated in json-like data:

```
{"key1": 0.32, "key2": 1000, "key3": 'text', ...}  
```

Data is captured with the help of a global thread object, then sent to a global queue.  An interval callback is fired every 1s to refresh the plots / readouts.
On the interval, data is pulled from the queue and written to an sqlite database file with (user specified filename & location).  The default file location is in a data/ directory next to the 'data_capure.py' script.  The filename defaults to include a timestamp so it is less likely to overwrite data on re-runs.

# Installation
clone the repository or download zip file, then use the requirements.txt file to install the requireed libraries.
```
git https://github.com/ohDAQ/ohDAQ-capture-serial.git <your directory>
pip3 install -r <your directory>/ohDAQ-capture-serial/requirements.txt
```
This installs with libraries on Rasbpian 10 (buster) when using the python 3.7.3 installed to the OS (or a venv derived from it).

# Usage
```
cd <your directory>/ohDAQ-capture-serial
python3 data_capture.py
```

Then open your browser (chromium on RPi) and enter the url: 'http://127.0.0.1:8050/'


# Example
In this example i have 2 thermocouples attached to breakouts with i2c MCP9600 devices which are in turn conencted to esp32 microcontroller which is writting the formatted data to serial.
data lines example:
```
{"s01_time_ms": 4139978, "s01_tempHJ_C": 2.000000e+01, "s01_tempCJ_C": 2.025000e+01, "s02_time_ms": 4139980, "s02_tempHJ_C": 1.906250e+01, "s02_tempCJ_C": 2.000000e+01}
{"s01_time_ms": 4140987, "s01_tempHJ_C": 1.993750e+01, "s01_tempCJ_C": 2.025000e+01, "s02_time_ms": 4140989, "s02_tempHJ_C": 1.906250e+01, "s02_tempCJ_C": 2.000000e+01}
{"s01_time_ms": 4141996, "s01_tempHJ_C": 1.993750e+01, "s01_tempCJ_C": 2.025000e+01, "s02_time_ms": 4141998, "s02_tempHJ_C": 1.881250e+01, "s02_tempCJ_C": 2.000000e+01}
{"s01_time_ms": 4143005, "s01_tempHJ_C": 1.993750e+01, "s01_tempCJ_C": 2.025000e+01, "s02_time_ms": 4143007, "s02_tempHJ_C": 1.912500e+01, "s02_tempCJ_C": 2.000000e+01}
```

's01_tempHJ_C' is the hot junction temperature for device 1   
's01_tempCJ_C' is the cold junction temperature for device 1  

In the example, I pinched the 2 hot junction ends and the temperature rose.

![](https://github.com/ohDAQ/public_gifs/blob/main/ohDAQ_usage.gif)



