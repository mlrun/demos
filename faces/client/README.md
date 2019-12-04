# Stream Video into iguazio using webcam

The following examples demonstrate ingesting video images into iguazio platform using web api 
from a webcam. 

clone demos to your laptop
* git clone https://github.com/mlrun/demos.git [mlrun demos](https://github.com/mlrun/demos) 

## go to client path
cd mlrun/demos/faces/client

## configure client init.ini file:
[app] <br>
log_level=info <br> 
partition = true <br>

[webapi] <br>
url = _**web api https url**_ <br>
container = _**container**_ <br>
stream_name = _**stream name**_ <br>

[nuclio] <br>
url = _**nuclio fucntion api**_ <br>
mount = /User <br>

[auth] <br>
username = _**user name**_ <br>
password = _**password**_ <br>
session_key = _**platform access key**_ <br>

## run the client
python VideoCapture.py

