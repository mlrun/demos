from mlrun import get_or_create_ctx
import os
import yaml

def configure_streamlit(ctx):
    '''
    opens the streamlit.yaml configuration file and edit necessary info.
    '''
    yaml_path = "/".join(os.environ['PYTHONPATH'].split("/")[:-1] + ['streamlit','streamlit.yaml'])
    ctx.logger.info(f"start working : {yaml_path}")
    ctx.logger.info(os.environ)
    with open(yaml_path) as f:
        streamlit_load = yaml.load_all(f,Loader = yaml.FullLoader)
        streamlit = []
        for i in streamlit_load:
            streamlit.append(i)
        ctx.logger.info("Editing streamlit configuration file")
        streamlit[0]["spec"]["template"]["spec"]["containers"][0]["env"][0]["value"] = os.environ["V3IO_ACCESS_KEY"]
        streamlit[0]["spec"]["template"]["spec"]["containers"][0]["env"][1]["value"] = "framesd:" + os.environ["FRAMESD_PORT_8081_TCP_PORT"]
        streamlit[0]["spec"]["template"]["spec"]["containers"][0]["env"][2]["value"] = "/".join([os.environ["V3IO_USERNAME"]] + streamlit[0]["spec"]["template"]["spec"]["containers"][0]["env"][2]["value"].split("/")[1:])
        streamlit[0]["spec"]["template"]["spec"]["volumes"][0]["flexVolume"]["options"]["accessKey"] = os.environ["V3IO_ACCESS_KEY"]
        streamlit[0]["spec"]["template"]["spec"]["volumes"][0]["flexVolume"]["options"]["subPath"] = "/" + os.environ["V3IO_USERNAME"]
        ctx.logger.info("Saving configuration file")
    with open(yaml_path,"w") as f:
        yaml.dump_all(streamlit,f,default_flow_style=False)

if __name__ == '__main__':
    ctx = get_or_create_ctx('streamlit-configuration')
    configure_streamlit(ctx)
