from conftest import (
    examples_path, has_secrets, here, out_path, tag_test, verify_state
)
from mlrun import NewTask, run_local, code_to_function
from mlrun import NewTask, get_run_db, new_function

base_spec = NewTask(params={'data_path': 'test/dataset/',
                            'frames_url': "https://framesd.default-tenant.app.vmdev22.lab.iguazeng.com",
                            'token': '5db1b7d1-f48f-4798-bed7-3c3d6f0767de',
                            'models_path': '../notebooks/functions/models.py'},
                    out_path=out_path
                    )


def test_encode_images():
    spec = tag_test(base_spec, 'test_run_local_encode_images')
    result = run_local(spec, command='../notebooks/functions/encode_images.py', workdir='./', artifact_path='./artifacts')
    verify_state(result)


def test_train():
    spec = tag_test(base_spec, 'test_run_local_train')
    result = run_local(spec, command='../notebooks/functions/train.py', workdir='./', artifact_path='./artifacts')
    verify_state(result)

