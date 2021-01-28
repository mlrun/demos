from streamlit import bootstrap

real_script = 'dashboard.py'

bootstrap.run(real_script, f'run.py {real_script}', [])