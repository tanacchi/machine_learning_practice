import cv2  # Actually, it is verision 4
import matplotlib.pyplot as plt
import time
import os
import curses

time_str = time.strftime("%YY_%mM_%dd_%Hh_%Mm_%Ss")
key_class_table = { 'p' : 'piece', 'f' : 'xxx', 'g': 'good'}
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
        stdscr.clear()
        image_count = class_num_table[image_class]
        class_num_table[image_class] += 1
        display_str = "[total:{}][class:{}] {} will be saved.".format(total_count, image_count, image_class)
        stdscr.addstr(0, 0, display_str)
        total_count += 1
        filename = images_root_dir + image_class + "/" + str(image_count) + ".jpg"
        cv2.imwrite(filename, frame)

    curses.endwin()
    camera.release()
    plt.close()

except KeyboardInterrupt:
    curses.endwin()
    camera.release()
    plt.close()
