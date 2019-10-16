import cv2  # Actually, it is verision 4
import matplotlib.pyplot as plt
import time
import os
import curses
import argparse


def make_key_class_table():
    def config_argparser():
        import string
        parser = argparse.ArgumentParser()
        for c in [ char for char in string.ascii_lowercase if char != 'h' ]:
            parser.add_argument('-' + c, help="key of class {}".format(c.upper()))
        return parser

    parser = config_argparser()
    args = parser.parse_args()
    args = vars(args)
    return { key : class_name for key, class_name in args.items() if class_name is not None }


def make_image_dirs(root_dir_path, class_names):
    for new_dir_name in class_names:
        new_dir_path = root_dir_path + "/" + new_dir_name
        os.makedirs(new_dir_path)


def show_progress(key_class_table, class_num_table, window, new_image_class=None):
    def format_progress_str(key, class_name, class_count, total_count):
        return "{} => {} (count: {}/{})".format(key, class_name, class_count, total_count)

    keys, names, counts = key_class_table.keys(), key_class_table.values(), class_num_table.values()
    key_name_count_list = [ (key, name, count) for key, name, count in zip(keys, names, counts) ]
    total_count = sum(counts)
    row_max = len(key_name_count_list)
    window.clear()
    for i in range(row_max):
        key, name, count = key_name_count_list[i]
        display_str = format_progress_str(key, name, count, total_count)
        window.addstr(i, 0, display_str)
    if new_image_class:
        window.addstr(row_max, 0, "{} was saved.".format(new_image_class))


def main(camera, stdscr):
    key_class_table = make_key_class_table()
    class_num_table = { key : 0 for key in key_class_table.values()}
    print(key_class_table)

    if key_class_table == {}:
        raise ValueError("More than 0 key & class_name are required.")

    images_root_dir = "./data/" + time.strftime("%YY_%mM_%dd_%Hh_%Mm_%Ss")
    make_image_dirs(root_dir_path=images_root_dir, class_names=key_class_table.values())

    show_progress(key_class_table, class_num_table, stdscr)
    while True:
        grab, frame = camera.read()
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb_image)
        plt.draw()
        plt.pause(0.01)

        curses.flushinp()
        key = stdscr.getch()
        key = chr(key)
        if not grab or not key in key_class_table:
            continue

        image_class = key_class_table[key]
        image_count = class_num_table[image_class]
        class_num_table[image_class] += 1

        filename = "{}/{}/{}.jpg".format(images_root_dir, image_class, str(image_count).zfill(8))
        cv2.imwrite(filename, frame)
        show_progress(key_class_table, class_num_table, stdscr, image_class)
        camera.grab()


if __name__ == '__main__':
    try:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        camera.set(cv2.CAP_PROP_FPS, 1)

        stdscr = curses.initscr()
        curses.noecho()
        plt.ion()

        main(camera, stdscr)
    finally:
        curses.endwin()
        camera.release()
        plt.close()
