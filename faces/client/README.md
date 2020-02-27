# Stream Video to the Platform Using a Webcam

The faces-demo client demonstrates ingestion of video images from a webcam to the Iguazio Data Science Platform ("the platform") using web APIs.
To run the client, do the following:

<a id="get-demos"></a>
## 1. Get the MLRun demos

Clone the [MLRun demos](https://github.com/mlrun/demos) repository (Git clone URL &mdash; https://github.com/mlrun/demos.git).

<a id="go-to-client-dir"></a>
## 2. Go to the demo's client directory

From a command-line shell, change directory to the **faces/client** directory in your local MLRun demos Git clone; for example:

```
cd mlrun/demos/faces/client
```

<a id="configure-client-init"></a>
## 3. Configure the client initialization

Edit the [**config/init.ini**](config/init.ini) file to configure the client initialization; replace the `<...>` placeholders to match your specific environment:

```ini
[app]
log_level = info

[nuclio]
url = <Nuclio-function API URL>
```

<a id="run-client"></a>
## 4. Run the client

Run **VideoCapture.py** Python client application:

```
python VideoCapture.py
```

