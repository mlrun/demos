from imutils import paths
import pandas as pd
import v3io_frames as v3f
import os
import shutil
import datetime

head = r"""

<!DOCTYPE html>

<html lang="en">

<head>

    <meta charset="UTF-8">

    <title>Dashboard</title>

    <link data-require="datatables@*" data-semver="1.10.12" rel="stylesheet" href="//cdn.datatables.net/1.10.20/css/jquery.dataTables.min.css" />

    <link data-require="font-awesome@*" data-semver="4.5.0" rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/font-awesome/4.5.0/css/font-awesome.css" />

    <link data-require="bootstrap-css@3.3.6" data-semver="3.3.6" rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/css/bootstrap.css" />

    <script data-require="jquery" data-semver="3.3.1" src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.0.0/jquery.js"></script>

    <script data-require="datatables@*" data-semver="1.10.20" src="//cdn.datatables.net/1.10.20/js/jquery.dataTables.min.js"></script>

</head>

<body>

"""

tbl = """<script> $(document).ready(function() {{

    var table = $(".dataframe").DataTable();
    var access_key = "{0}"

    $('.dataframe tbody')
        .on('mouseenter', 'td', function () {{
            var colIdx = table.cell(this).index().column;

            $(table.cells().nodes()).removeClass('highlight');
            $(table.column(colIdx).nodes()).addClass('highlight');
        }})
        .on('click', 'tr', function (e) {{
            e.stopPropagation();
            if ($(this).hasClass('selected')) {{
                $(this).removeClass('selected');
            }}
            else {{
                table.$('tr.selected').removeClass('selected');
                var data = table.row($(this)).data()
                $(this).addClass('selected');
                var xhr = new XMLHttpRequest();
                var rel_url = "{1}" + data[4].slice(6)
                xhr.open("GET", rel_url);
                xhr.setRequestHeader("x-v3io-session-key", access_key)
                xhr.responseType = "blob";
                xhr.onload = function(){{
                    $('#browsed').attr("src", window.URL.createObjectURL(xhr.response))
                }};
                xhr.send();
            }}
        }});

    $('#button').click( function () {{
        table.row('.selected').remove().draw(false);
    }});
}}); 

</script>"""

img_tag = "<img src = '' alt = 'Please select image to show' id = 'browsed' height = 750 width = 750 align = 'middle'> </img>"

client = v3f.Client("framesd:8081", container="users")

access_key = os.environ['V3IO_ACCESS_KEY']
web_api_prefix = os.environ['WEB_API_PREFIX']

def load_images(data_path):
    return [f for f in paths.list_images(data_path) if '.ipynb' not in f]


def load_enc_df():
    return client.read(backend="kv", table='iguazio/demos/demos/realtime-face-recognition/artifacts/encodings', reset_index=True)


def handler(context, event):

    context.logger.info("generating df")

    enc_df = load_enc_df()
    view_df = enc_df[['fileName', 'camera', 'time', 'imgUrl']]
    view_df = view_df.rename(columns={'fileName': 'identifier'})

    browse_tbl_html = view_df.to_html()

    html = head + tbl.format(access_key, web_api_prefix) + browse_tbl_html + img_tag + '</body></html>'

    return context.Response(body=html, headers={}, content_type='text/html', status_code=200)
