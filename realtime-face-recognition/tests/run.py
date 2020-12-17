from streamlit import bootstrap

real_script = '../streamlit/dashboard.py'

bootstrap.run(real_script, f'run.py {real_script}', [])