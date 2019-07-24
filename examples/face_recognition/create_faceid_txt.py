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


def create_txt(dir_root):
    faceid_list_str = ""
    dir_list = read_dir(dir_root)
    count_dir = 0
    count_img = 0
    for item in dir_list:
        path_ = os.path.join(dir_root, item)
        img_list = read_file_name(path_)
        for img_name in img_list:
            faceid_list_str += os.path.join(path_, img_name) + " " + item + "\n"
            count_img += 1
            if count_img % 1000 == 0:
                print("get %d images..." % count_img)
        count_dir += 1
        if count_dir % 500 == 0:
            print("finished %d/%d" % (count_dir, len(dir_list)))
    print("create txt done! %d dirs, %d images" % (count_dir, count_img))
    return faceid_list_str


def main():
    face_root_dir = "/home/ubuntu/disk_d/tanghy/face/imgs"
    faceid_txt_path = "./faceid_list.txt"
    faceid_str = create_txt(face_root_dir)
    with open(faceid_txt_path, "w") as fp:
        fp.write(faceid_str)


if __name__ == '__main__':
    main()

