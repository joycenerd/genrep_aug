from os import listdir
from os.path import isfile, join, basename, isfile
import numpy as np

def make_html(home_dir):
    curr_dir = home_dir.split('/')[-1]
    std_list = ['1', '2', '3', '4', '5']
    try:
        index_html_path = join(home_dir, "index.html")
        print('Saving index file in', index_html_path)
        fid = open(index_html_path, 'w', encoding = 'utf-8')

        fid.write('<table style="text-align:center;">')
        header_str = '<tr><td>anchore filename</td><td>anchore</td>'
        for i in range(len(std_list)):
            header_str += '<td>'+std_list[i]+'</td>'
        header_str += '</tr>'
        fid.write(header_str)
        home_dir = join(home_dir, 'train')
        dir_list = listdir(home_dir)   
        for ii in range(len(dir_list)): 
        # for ii in range(10):
            images_dir = join(home_dir, dir_list[ii])
            if isfile(images_dir):
                continue
            file_names = [f for f in listdir(images_dir) if join(images_dir, f).endswith('_anchor.png')]

            # for i in range(len(file_names)):
            for i in range(3):
                fid.write('<tr>')
                anchor_file_name = join('train', basename(images_dir), file_names[i])
                fid.write('<td>' + anchor_file_name + '</td>')
                anchor_file_name = join('train', basename(images_dir), file_names[i])
                fid.write('<td><a href="' + anchor_file_name + '"><img src="' +
                        anchor_file_name + '"/></a></td>')
                for j in range(len(std_list)):
                    if std_list[j] == '1':
                        std_file_name = anchor_file_name.replace('anchor', '1.0_1')
                    elif std_list[j] == '2':
                        std_file_name = anchor_file_name.replace('anchor', '1.0_2')
                    elif std_list[j] == '3':
                        std_file_name = anchor_file_name.replace('anchor', '1.0_3')
                    elif std_list[j] == '4':
                        std_file_name = anchor_file_name.replace('anchor', '1.0_4')
                    elif std_list[j] == '5':
                        std_file_name = anchor_file_name.replace('anchor', '1.0_5')
                        
                    fid.write('<td><a href="' + std_file_name + '"><img src="' +
                        std_file_name + '"/></a></td>')
                fid.write('</tr>')
        fid.write('</table>')

    finally:
        fid.close()

make_html('/data/scratch/jahanian/ganclr_results_2/biggan256tr0.4-png_rnd_gauss_angular-std1.0_100_5_samples')
