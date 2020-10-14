from conftest import (
    examples_path, has_secrets, here, out_path, tag_test, verify_state
)
from mlrun import NewTask, run_local, code_to_function
from mlrun import NewTask, get_run_db, new_function

base_spec = NewTask(params={'data_path': 'test/dataset/'},
                    out_path=out_path
                    )


def test_encode_images():
    spec = tag_test(base_spec, 'test_run_local_parquet')
    result = run_local(spec, command='../functions/encode_images.py', workdir='./', artifact_path='./artifacts')
    verify_state(result)
