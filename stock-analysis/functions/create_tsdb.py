import os
import v3io_frames as v3f

frames_uri = 'https://framesd.default-tenant.app.dev39.lab.iguazeng.com'
client = v3f.Client(frames_uri, container=os.getenv('V3IO_CONTAINER', 'bigdata'),token='258dfd92-f43c-4934-aa08-611c3d38e336')
client.create(backend='tsdb', table='stocks/stocks_tsdb', rate='1/s', if_exists=1)