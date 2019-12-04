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
url = <https://webapi.default-tenant.app.com> <br>
container = _container_ <br>
stream_name = _stream name_ <br>

[nuclio] <br>
url = _nuclio_fucntion_api_ <br>
mount = /User <br>

[auth] <br>
username = _user name_ <br>
password = _password_ <br>
session_key = _platform access key_ <br>

## run the client
python VideoCapture.py

