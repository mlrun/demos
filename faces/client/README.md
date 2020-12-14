# Faces Client

The faces client uses open-cv to stream images into iguazio platform.
the client uses the laptop camera to send frames into the pre-deployed nuclio serving function. 

<a id="get-mlrun-demos"></a>
## Clone the MLRun Demos

Clone the [mlrun/demos](https://github.com/mlrun/demos) repository into your laptop (Git clone URL &mdash; https://github.com/mlrun/demos.git).
it is required in order to use your laptop camera and stream its frames into the iguazio system 

<a id="go-to-client-dir"></a>
## Go to the Client Directory

```sh
cd mlrun/demos/faces/client
```

<a id="config-client-init"></a>
## Configure the Client Initialization

Edit the [**config/init.ini**](config/init.ini) client-initialization file to configure the client initialization; replace the `<...>` placeholders to match your specific environment.
set the nuclio endpoint into nuclio-api-serving-function 
the function can be found under Projects(tab)->faces->

```ini
[app]
log_level=info

[nuclio]
url = <nuclio endpoint>
```

<a id="run-client"></a>
## Run the Client

```sh
python video_capture.py
```

unless you are a famous actor the expected response from the client is :
 [{"coords": [75, 889, 259, 706], "name": "unknown", "label": -1, "confidence": 0.3509389671540804, "encoding": [-0.20812971889972687, 0.13344672322273254, ....]}]'



