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
        for c in string.ascii_lowercase:
            if c != 'h':
                parser.add_argument('-' + c, help="key of class {}".format(c.upper()))
        return parser

    parser = config_argparser()
    args = parser.parse_args()
    args = vars(args)
    return { key : class_name for key, class_name in args.items() if class_name is not None }


key_class_table = make_key_class_table()
print(key_class_table)

if key_class_table == {}:
    raise ValueError("More than 0 key & class_name are required.")

time_str = time.strftime("%YY_%mM_%dd_%Hh_%Mm_%Ss")
class_num_table = { key : 0 for key in key_class_table.values()}

images_root_dir = "./data/" + time_str + "/"
for new_dir_name in key_class_table.values():
    new_dir_path = images_root_dir + new_dir_name + "/"
    os.makedirs(new_dir_path)

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
camera.set(cv2.CAP_PROP_FPS, 1)

plt.ion()

stdscr = curses.initscr()
curses.noecho()

total_count = 0
try:
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
        total_count += 1

        stdscr.clear()
        display_str = "[total:{}][class:{}] {} will be saved.".format(total_count, image_count, image_class)
        stdscr.addstr(0, 0, display_str)

        filename = images_root_dir + image_class + "/" + str(image_count).zfill(8) + ".jpg"
        cv2.imwrite(filename, frame)

    curses.endwin()
    camera.release()
    plt.close()

except KeyboardInterrupt:
    curses.endwin()
    camera.release()
    plt.close()
