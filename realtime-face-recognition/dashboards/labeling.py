from imutils import paths
import pandas as pd
import os

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

tbl = """<script> 
    
    $(document).ready(function() {{
        var access_key = "{0}"
        var table = $(".dataframe").DataTable();
        var options_table = $("#options").DataTable();
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
                    var rel_url = "{1}" + data[1].slice(6)
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
    
    
         $('#options tbody')
            .on('mouseenter', 'td', function () {{
                var colIdx = options_table.cell(this).index().column;
    
                $(options_table.cells().nodes()).removeClass('highlight');
                $(options_table.column(colIdx).nodes()).addClass('highlight');
            }})
            .on('click', 'tr', function (e) {{
                e.stopPropagation();
                if ($(this).hasClass('selected')) {{
                    $(this).removeClass('selected');
                }}
                else {{
                    options_table.$('tr.selected').removeClass('selected');
                    var data = options_table.row($(this)).data()
                    $(this).addClass('selected');
                }}
            }});
    }});
    
    function tagSelected(){{
        var src_img_url = "{1}" + $(".dataframe tr.selected td").text().split("/").slice(2).join("/")
        var dst_img_url = "{1}" + $(".dataframe tr.selected td").text().split("/").slice(2,6).join('/') + "/input/" + $('#options tr.selected td').text().replace(" ", "_") + "/" + $(".dataframe tr.selected td").text().split("/").slice(-1).join("/")
        var access_key = "{0}"
        
        var xhr_get = new XMLHttpRequest();            
        xhr_get.open("GET", src_img_url);
        xhr_get.setRequestHeader("x-v3io-session-key", access_key);
        xhr_get.responseType = "blob";
        xhr_get.onload = function(){{
            var xhr_post = new XMLHttpRequest();
            xhr_post.open('POST', dst_img_url);
            xhr_post.setRequestHeader("x-v3io-session-key", access_key);
            xhr_post.onload = function(){{
                var xhr_delete = new XMLHttpRequest();
                xhr_delete.open('DELETE', src_img_url)
                xhr_delete.setRequestHeader("x-v3io-session-key", access_key);
                xhr_delete.onload = alert("Person successfully tagged as " + $('#options tr.selected td').text());
                xhr_delete.send();
            }};
            xhr_post.send(xhr_get.response);
        }};
        xhr_get.send();
    }}
        
    function addEmployee(){{
        var new_user = prompt("Please provide name of the employee in the image")
        new_user = new_user.split(" ").join("_")
        var src_img_url = "{1}" + $(".dataframe tr.selected td").text().split("/").slice(2).join("/")
        var dst_folder = "{1}" + $(".dataframe tr.selected td").text().split("/").slice(2,6).join('/') + "/input/" + new_user
        var dst_img_url = dst_folder + "/" + $(".dataframe tr.selected td").text().split("/").slice(-1).join("/")
        var access_key = "{0}"
        
        var xhr_get = new XMLHttpRequest();            
        xhr_get.open("GET", src_img_url);
        xhr_get.setRequestHeader("x-v3io-session-key", access_key);
        xhr_get.responseType = "blob";
        xhr_get.onload = function(){{
            var xhr_post = new XMLHttpRequest();
            xhr_post.open('POST', dst_img_url);
            xhr_post.setRequestHeader("x-v3io-session-key", access_key);
            xhr_post.onload = function(){{
                var xhr_delete = new XMLHttpRequest();
                xhr_delete.open('DELETE', src_img_url)
                xhr_delete.setRequestHeader("x-v3io-session-key", access_key);
                xhr_delete.onload = alert("Person successfully tagged as " + new_user.split("_").join(" "));
                xhr_delete.send();
            }};
            xhr_post.send(xhr_get.response);
        }};
        xhr_get.send();
    }}
        
    function notEmployed(){{
        var src_img_url = "{1}" + $(".dataframe tr.selected td").text().split("/").slice(2).join("/") 
        var dst_img_url = "{1}" + $(".dataframe tr.selected td").text().split("/").slice(2,6).join('/') + "/unknowns/" + $(".dataframe tr.selected td").text().split("/").slice(-1).join("/")
        var access_key = "{0}"
        
        var xhr_get = new XMLHttpRequest();            
        xhr_get.open("GET", src_img_url);
        xhr_get.setRequestHeader("x-v3io-session-key", access_key);
        xhr_get.responseType = "blob";
        xhr_get.onload = function(){{
            var xhr_post = new XMLHttpRequest();
            xhr_post.open('POST', dst_img_url);
            xhr_post.setRequestHeader("x-v3io-session-key", access_key);
            xhr_post.onload = function(){{
                var xhr_delete = new XMLHttpRequest();
                xhr_delete.open('DELETE', src_img_url)
                xhr_delete.setRequestHeader("x-v3io-session-key", access_key);
                xhr_delete.onload = alert("Person successfully tagged as unknown");
                xhr_delete.send();
            }};
            xhr_post.send(xhr_get.response);
        }};
        xhr_get.send();
    }}
</script>"""

img_tag = "<img src = '' alt = 'Please select image to show' id = 'browsed' height = 750 width = 750 align = 'middle'> </img>\n\n"

apply_btns = """<button type=\"button\" onclick=\"tagSelected()\">Tag as selected employee</button> 
                <button type=\"button\" onclick=\"addEmployee()\">Add new employee</button>
                <button type=\"button\" onclick=\"notEmployed()\">Tag as unknown</button>"""

access_key = os.environ['V3IO_ACCESS_KEY']
web_api_prefix = os.environ['WEB_API_PREFIX']


def list_to_html_table(lol, table_id, table_head):
    head_cells = ""
    for h in table_head:
        head_cells += f"<th>{h}</th>"
    head_row = f"<tr>{head_cells}</tr>"
    head = f"<thead>{head_row}</thead>"
    body = ""
    for l in lol:
        cells = ""
        for c in l:
            cells += f"<td>{c}</td>"
        row = f"<tr>{cells}</tr>\n\t"
        body += row

    meta = f"<table id=\"{table_id}\">\n\t{head}<tbody>\n\t{body}\n</tbody>\n</table>"
    return meta


def handler(context, event):

    data_path = '/User/demos/demos/realtime-face-recognition/dataset/'
    artifact_path = 'User/demos/demos/realtime-face-recognition/artifacts/'

    classes_url = artifact_path + 'idx2name.csv'
    classes_df = pd.read_csv(classes_url)

    known_classes = [n.replace('_', ' ') for n in classes_df['name'].values if 'MACOSX' not in n]
    images = [f for f in paths.list_images(data_path + 'label_pending') if '.ipynb' not in f]

    d = {'url': images}
    paths_df = pd.DataFrame(d)
    paths_tbl_html = paths_df.to_html()

    options_html = list_to_html_table([[o] for o in known_classes], "options", ["Choose option:"])

    html = head + tbl.format(access_key, web_api_prefix) + paths_tbl_html + "\n" + img_tag + options_html + apply_btns + '</body></html>'

    return context.Response(body=html, headers={}, content_type='text/html', status_code=200)
