# Stream Video into iguazio using webcam

The following examples demonstrate ingesting video images into iguazio platform using web api 
from a webcam. 

clone demos to your laptop
* git clone https://github.com/mlrun/demos.git [mlrun demos](https://github.com/mlrun/demos) 

## go to client path
cd mlrun/demos/faces/client

## configure client:
[app] <br>
log_level=info <br> 
partition = true <br>

[webapi] <br>
url = <https://webapi.default-tenant.app.com> <br>
container = container <br>
stream_name = stream <br>

[nuclio] <br>
url = nuclio_fucntion_api <br>
mount = /User <br>

[auth] <br>
username = user name <br>
password = password <br>
session_key = session key <br>

## run the client
python VideoCapture

