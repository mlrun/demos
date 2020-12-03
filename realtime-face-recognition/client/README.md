# Image-Streaming Client

[Overview](#overview)&nbsp;| [Get the MLrun Demos](#get-mlrun-demos)&nbsp;| [Go to the Client Directory](#go-to-client-dir)&nbsp;| [Configure the Client Initialization](#config-client-init)&nbsp;| [Run the Client](#run-client)

## Overview

The faces-demo client demonstrates ingestion of video images from a webcam to the Iguazio Data Science Platform ("the platform") using web APIs.

<a id="get-mlrun-demos"></a>
## Get the MLRun Demos

Clone the [mlrun/demos](https://github.com/mlrun/demos) repository (Git clone URL &mdash; https://github.com/mlrun/demos.git).

<a id="go-to-client-dir"></a>
## Go to the Client Directory

```sh
cd mlrun/demos/realtime-face-recognition/client
```

<a id="config-client-init"></a>
## Configure the Client Initialization

Edit the [**config/init.ini**](config/init.ini) client-initialization file to configure the client initialization; replace the `<...>` placeholders to match your specific environment.

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

