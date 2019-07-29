# -*- coding: utf-8 -*-

import os
from random import shuffle


def read_file_name(dir_path):
    file_name = []
    file_lsit = os.listdir(dir_path)
    for item in file_lsit:
        if os.path.isfile(os.path.join(dir_path, item)):
            file_name.append(item)
    return file_name


def read_dir(dir_root):
    dir_name = []
    name_list = os.listdir(dir_root)
    count = 0
    for item in name_list:
        if os.path.isdir(os.path.join(dir_root, item)):
            try:
                _ = int(item)
                dir_name.append(item)
            except Exception as _:
                print("%s not a number" % item)
        count += 1
        if count % 500 == 0:
            print("finished %d/%d" % (count, len(name_list)))
    print("read dir done! %d dirs" % count)
    return dir_name


def get_all_data(dir_root, is_shuffle=True):
    faceid_list = []
    dir_list = read_dir(dir_root)
    count_dir = 0
    count_img = 0
    for item in dir_list:
        path_ = os.path.join(dir_root, item)
        img_list = read_file_name(path_)
        for img_name in img_list:
            faceid_list.append([os.path.join(path_, img_name), item])
            count_img += 1
            if count_img % 10000 == 0:
                print("get %d images..." % count_img)
        count_dir += 1
        if count_dir % 1000 == 0:
            print("finished %d/%d" % (count_dir, len(dir_list)))
    print("create txt done! %d dirs, %d images" % (count_dir, count_img))
    if is_shuffle:
        print("shuffle data...")
        shuffle(faceid_list)
    return faceid_list


def get_small_data(dir_root, num=100, is_shuffle=True):
    faceid_list = []
    dir_list = read_dir(dir_root)
    dir_list = [int(item) for item in dir_list]
    dir_list = sorted(dir_list)[:num]
    dir_list = [str(item) for item in dir_list]
    count_dir = 0
    count_img = 0
    for item in dir_list:
        path_ = os.path.join(dir_root, item)
        img_list = read_file_name(path_)
        for img_name in img_list:
            faceid_list.append([os.path.join(path_, img_name), item])
            count_img += 1
            if count_img % 1000 == 0:
                print("get %d images..." % count_img)
        count_dir += 1
    print("create txt done! %d dirs, %d images" % (count_dir, count_img))
    if is_shuffle:
        print("shuffle data...")
        shuffle(faceid_list)
    return faceid_list


def list2str(faceid_list):
    temp_list = [None]*len(faceid_list)
    for i in range(len(faceid_list)):
        temp_list[i] = ' '.join(faceid_list[i])
    faceid_str = '\n'.join(temp_list)
    return faceid_str


def main():
    face_root_dir = "/home/ubuntu/disk_d/tanghy/face/imgs"
    train_faceid_txt_path = "./train_faceid_list.txt"
    test_faceid_txt_path = "./test_faceid_list.txt"
    faceid_list = get_all_data(face_root_dir, is_shuffle=True)
    split_ = 10000
    train_str = list2str(faceid_list[:-split_])
    test_str = list2str(faceid_list[-split_:])
    with open(train_faceid_txt_path, "w") as fp:
        fp.write(train_str)
    with open(test_faceid_txt_path, "w") as fp:
        fp.write(test_str)

    train_small_faceid_txt_path = "./train_small_faceid_list.txt"
    test_small_faceid_txt_path = "./test_small_faceid_list.txt"
    faceid_list_small = get_small_data(face_root_dir, num=100, is_shuffle=True)
    split_small = 500
    train_str = list2str(faceid_list_small[:-split_small])
    test_str = list2str(faceid_list_small[-split_small:])
    with open(train_small_faceid_txt_path, "w") as fp:
        fp.write(train_str)
    with open(test_small_faceid_txt_path, "w") as fp:
        fp.write(test_str)


if __name__ == '__main__':
    main()

