# -*- coding: utf-8 -*-

import os


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
    for item in name_list:
        if os.path.isdir(os.path.join(dir_root, item)):
            try:
                _ = int(item)
                dir_name.append(item)
            except Exception as _:
                print("%s not a number" % item)
    return dir_name


def create_txt(dir_root):
    faceid_list_str = ""
    dir_list = read_dir(dir_root)
    for item in dir_list:
        path_ = os.path.join(dir_root, item)
        img_list = read_file_name(path_)
        for img_name in img_list:
            faceid_list_str += os.path.join(path_, img_name) + " " + item + "\n"
    return faceid_list_str


def main():
    face_root_dir = "/home/ubuntu/disk_d/tanghy/face/imgs"
    faceid_txt_path = "./faceid_list.txt"
    faceid_str = create_txt(face_root_dir)
    with open(faceid_txt_path, "w") as fp:
        fp.write(faceid_str)


if __name__ == '__main__':
    main()

