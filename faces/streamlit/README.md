# Deployment of streamlit dashboard for labeling images and browsing collected data

This markdown presents instructions for deploying streamlit dashboard to our application needs:

* Edit demos/faces/streamlit/streamlit.yaml with your platform access key in the two following locations:

```
containers:
      .
      .
      .
        env:
        - name: "V3IO_ACCESS_KEY"
          value: <ACCESS_KEY>
      .
      .
      .
```
and

```
volumes:
     - flexVolume:
         driver: v3io/fuse
         options:
          accessKey: <ACCESS_KEY>
          container: users
          subPath: /iguazio
       name: fs
```

* Ensure that you have already have a trained model generated through [face-recognition.ipynb](https://github.com/mlrun/demos/faces/notebooks/face-recognition.ipynb)
* Verify that dashboard.py script exists under the path "/User/demos/demos/faces/streamlit/dashboard.py", the file should be there upon git cloning mlrun/demos in jupyter under ~/demos.
* Deploy streamlit as a service from your edited streamlit.yaml in iguazio platform using the following command: `kubectl -n default-tenant apply -f streamlit.yaml`
* In your web browser go to **your_app_node_ip_address**:30090 and view your interactive streamlit dashboard:

<br><p align="center"><img src="dashboard.png" width="1000" height="500"/></p><br>
