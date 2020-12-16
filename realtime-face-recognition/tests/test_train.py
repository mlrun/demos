import v3io_frames as v3f
from functions.train import read_encodings_table

FRAMES_URL = 'https://framesd.default-tenant.app.vmdev22.lab.iguazeng.com'
TOKEN = '5db1b7d1-f48f-4798-bed7-3c3d6f0767de'


def test_train():
    client = v3f.Client(FRAMES_URL, container="faces",token=TOKEN)
    encodings_df = read_encodings_table(client, "encodings")
    print(encodings_df)
