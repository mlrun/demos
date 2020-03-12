# Stream Video into iguazio using webcam

The following examples demonstrate ingesting video images into iguazio platform using web api 
from a webcam. 

clone demos to your laptop
* git clone https://github.com/mlrun/demos.git [mlrun demos](https://github.com/mlrun/demos) 

## go to client path
cd mlrun/demos/faces/client

## configure client init.ini file:
### edit bold variables

[app] <br>
log_level=info <br> 

[nuclio] <br>
url = _**nuclio fucntion api**_ <br>

## run the client
python VideoCapture.py

