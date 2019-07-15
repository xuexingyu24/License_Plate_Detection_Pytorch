import os
import sys
sys.path.append(os.getcwd())
import assemble

pnet_postive_file = 'anno_store/pos_12_val.txt'
pnet_part_file = 'anno_store/part_12_val.txt'
pnet_neg_file = 'anno_store/neg_12_val.txt'
# pnet_landmark_file = './anno_store/landmark_12.txt'
imglist_filename = 'anno_store/imglist_anno_12_val.txt'

if __name__ == '__main__':

    anno_list = []

    anno_list.append(pnet_postive_file)
    anno_list.append(pnet_part_file)
    anno_list.append(pnet_neg_file)
    # anno_list.append(pnet_landmark_file)

    chose_count = assemble.assemble_data(imglist_filename ,anno_list)
    print("PNet train annotation result file path:%s" % imglist_filename)
