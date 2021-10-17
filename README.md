# Vigil
An AI Social Distancing Detection Tool  
Note: In order to run this program, Yolov3 weights must be installed at: https://pjreddie.com/darknet/yolo/

## What it does 
Vigil is an AI Social Distancing Detection Tool that helps to enforce COVID-19 Safe Distancing Measures. Given any video feed, Vigil helps to identify individuals who have gathered in groups larger than the permissible size and alerts the user about the rule flouters. 
This eliminated the need for safe distancing ambassadors to patrol crowded areas as a single Safe Distancing Manager can monitor these areas with the help of surveillance cameras. Managers can then deploy their ambassadors to warn rule flouters upon alerts by Vigil.  


## How we built it
We built Vigil using the OpenCV library and the pre-trained object detection model, Yolov3. We conducted multiple tests to fine-tune Vigil, allowing it to accurately identify instances of close contact.


## What Inspired Us 
As the COVID-19 pandemic spread across the world, many countries implemented Social Distancing Measures (SDM)  to tackle the infectivity of the virus. In Singapore, Social Distancing Ambassadors (SDA) have been introduced to help enforce these rules and warn those who fail to adhere to them. Currently, these SDAs have to patrol crowded areas and manually identify SDM breaches. 
NDSM feels that this is an inefficient use of manpower as too many SDAs are required for the enforcement of SDMs, especially as Singapore pivots to treat COVID-19 as endemic. Furthermore, SDAs also face constant exposure risk as their job requires them to patrol areas with high human traffic. 
Thus, we decided to build an object detection tool, which can aid SDAs in their everyday job, and serves as a central management system at malls. 


## Challenges Faced
When starting this project, we knew nothing about machine learning or deep convolutional neural network. While googling and researching, we also stumbled upon code that we did not understand and models that were unsuitable for our use. It took us quite some time to find a suitable object detection model called Yolov3. 
Further, documentation for Yolov3 was not very comprehensive, so we had to figure out how to implement it and which parameters to change to get the program working. 

