camera = 3

# Перед запуском установить библиотеки
# pip install ultralytics
# pip install supervision
# pip install lap

# Подсчет людей внутри зоны
# Смысл алгоритма: отметил на кадре 2 зоны, при попадании объекта внутрь зоны, запоминаем его на несколько кадров.
# Если объект перешел в другую зону, удаляю его из запомненых и увеличиваю счетчик входящих/выходящих
#
import cv2
import random
from ultralytics import YOLO
import supervision as sv
import numpy as np
import configparser
import os
from datetime import datetime


def create_config(path):
    """
    Create a config file
    """
    config = configparser.ConfigParser()
    config.add_section("camera1")
    config.set("camera1", "RTSP_URL", "rtsp://stream:ShAn675sb5@31.173.67.209:13554")
    config.set("camera1", "Out", "391, 1; 450, 420; 950, 350; 838, 1")
    config.set("camera1", "In", "398, 720; 450, 420; 950, 350; 1263, 720")
    config.set("camera1", "deep", "3")
    config.add_section("camera2")
    config.set("camera2", "RTSP_URL", "rtsp://stream:ShAn675sb5@178.176.43.250:13554")
    config.set("camera2", "Out", "500, 1; 500, 550; 1000, 600; 1100, 1")
    config.set("camera2", "In", "300, 1079; 500, 550; 1000, 600; 1300, 1079")
    config.set("camera2", "deep", "3")
    config.add_section("camera3")
    config.set("camera3", "RTSP_URL", "rtsp://stream:ShAn675sb5@188.170.35.150:13554")
    config.set("camera3", "Out", "400, 1; 550, 800; 1400, 400; 1000, 1")
    config.set("camera3", "In", "400, 1079; 550, 800; 1400, 400; 1800, 1079")
    config.set("camera3", "deep", "3")

    with open(path, "w") as config_file:
        config.write(config_file)


def get_config(path):
    """
    Returns the config object
    """
    if not os.path.exists(path):
        create_config(path)

    config = configparser.ConfigParser()
    config.read(path)
    return config


def get_setting(path, section, setting):
    """
    Print out a setting
    """
    config = get_config(path)
    value = config.get(section, setting)
    msg = "{section} {setting} is {value}".format(
        section=section, setting=setting, value=value
    )

    print(msg)
    return value


def update_setting(path, section, setting, value):
    """
    Update a setting
    """
    config = get_config(path)
    config.set(section, setting, value)
    with open(path, "w") as config_file:
        config.write(config_file)
        print('saved settings', config)


def inPolygon(x, y, crd):  # Функция для определения лежит ли точка внутри многоугольника
    c = 0
    for i in range(len(crd)):
        if (((crd[i][1] <= y and y < crd[i - 1][1]) or (crd[i - 1][1] <= y and y < crd[i][1])) and
                (x > (crd[i - 1][0] - crd[i][0]) * (y - crd[i][1]) / (crd[i - 1][1] - crd[i][1]) + crd[i][
                    0])): c = 1 - c
    return c


def inLabel(p, lb, sl):  # выдает количество меток во всех полигонах кроме "sl"
    ret = 0
    for i in range(len(lb)):
        if i != sl:
            for j in range(len(lb[i])):
                ret += lb[i][j].count(p)
    return ret


def delFromLabel(p, lb):  # удаляет метку из массива
    for i in range(len(lb)):
        for j in range(len(lb[i])):
            for k in range(lb[i][j].count(p)):
                lb[i][j].remove(p)

def flatten_and_sort(lst):
    flattened_list = [item for sublist in lst for item in sublist]  # Разворачиваем список в одномерный список
    unique_sorted_list = sorted(set(flattened_list))  # Удаляем дубликаты и сортируем список
    return unique_sorted_list

def print_labels_obj(s=''):
    print(s)
    for sl in range(deep):
        print('kadr', sl - deep + 1)
        # print(*[i for i in range(polygons)],sep='\t')
        maxp = max([len(labels_obj[sl][k]) for k in range(polygons)])
        for no in range(maxp):
            for pl in range(polygons):
                if no <= len(labels_obj[sl][pl]) - 1:
                    print(labels_obj[sl][pl][no], end='\t')
            print()


def mouse_events(event, x, y, flags, param):
    global lbd, mx, my, coord, points
    if lbd and event == cv2.EVENT_MOUSEMOVE:
        mx = x
        my = y
        print(lbd, x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        # qq = points[0].index([90, 84])
        if ([int(x / radius), int(y / radius)] in points[0]) or ([int(x / radius), int(y / radius)] in points[1]):
            if ([int(x / radius), int(y / radius)] in points[0]):
                coord = [0, points[0].index([int(x / radius), int(y / radius)])]
            else:
                coord = [1, points[1].index([int(x / radius), int(y / radius)])]
            lbd = True
            mx = x
            my = y
        print(lbd, x, y, coord)
    if event == cv2.EVENT_LBUTTONUP:
        lbd = False
        mx = x
        my = y
        points = [[[int(x / radius), int(y / radius)] for x, y in i] for i in POLYGON_COORDS]


# Считываем сохраненные параметры
ini_file = 'settings.ini'
# URL камеры
RTSP_URL = get_setting(ini_file, 'camera' + str(camera), 'RTSP_URL')
# координаты вершин полигона, который мы будем использовать для подсчета проходящих объектов
POLYGON_COORDS = [
    [list(map(int, ss.split(','))) for ss in get_setting(ini_file, 'camera' + str(camera), 'Out').split(';')],
    [list(map(int, ss.split(','))) for ss in get_setting(ini_file, 'camera' + str(camera), 'In').split(';')]]
# на сколько кадров запоминать объект внутри полигона
deep = int(get_setting(ini_file, 'camera' + str(camera), 'deep'))

edit = False  # Режим редактирования полигонов
radius = 10  # Загрубление координат в radius раз, для поиска вершин полигона
points = [[[int(x / radius), int(y / radius)] for x, y in i] for i in POLYGON_COORDS]
isClosed = True  # полигон, а не полилиния
color = (255, 0, 0)  # цвет полигона
thickness = 2  # толщина линии полигона
lbd = False  # левая кнопка мыши нажата
mx = 0  # координаты курсора мыши, когда левая кнопка мыши нажата
my = 0
coord = [0, 0]
save_key = False  # Режим сохранения кадров
save_video_key = False  # Режим сохранения видео
saved_kadr = -100
txt_path = 'txt/'
img_path = 'img/'
vid_path = 'video/'
if not os.path.exists(txt_path): os.mkdir(txt_path)
if not os.path.exists(img_path): os.mkdir(img_path)
if not os.path.exists(vid_path): os.mkdir(vid_path)
polygons = len(POLYGON_COORDS)
persons = [0 for _ in range(polygons)]  # счетчик входящих/выходящих
# Создаем экземпляр модели YOLO
model = YOLO("yolov8s.pt")
model = YOLO('yolov8s_modN.pt')  # load a custom model

box_annotator = sv.BoxAnnotator(
    thickness=2,
    text_thickness=1,
    text_scale=0.5
)
box_annotator_c = sv.BoxAnnotator(
    thickness=2,
    text_thickness=1,
    text_scale=0.5,
    color=sv.Color.white()
)
kadr = 0

# создаем структуру для запоминания объектов внутри полигонов
labels_obj = [[[-1 for i in range(1)] for k in range(deep)] for l in
              range(polygons)]
# 1 количество полигонов
# 2 Глубина - сколько кадров храним данные
# 3 Метки объектов

lf = open('camera' + str(camera) + '.log', "a")  # log file
# Итерируемся по кадрам видео
for result in model.track(source=RTSP_URL, show=False, stream=True, agnostic_nms=True,
                          classes=0, iou=0.7, conf=0.25,
                          tracker="botsort.yaml"):  # tracker="bytetrack.yaml",tracker="botsort.yaml"
    kadr += 1
    # Получаем текущий кадр и результаты детекции, приводим к одному размеру
    frame = result.orig_img
    img = frame.copy()
    detections = sv.Detections.from_yolov8(result)
    # Присваиваем каждому объекту ID, чтобы отслеживать его в последующих кадрах
    if result.boxes.id is not None:
        detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

    # Оставляем только объекты, соответствующие нужным классам
    detections = detections[(detections.class_id == 0)]
    dt = detections.xyxy.copy()
    # Формируем метки для объектов

    labels = [
        f"{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
        for _, confidence, class_id, tracker_id
        in detections
    ]

    # Аннотируем прямоугольными рамками объекты на кадре
    frame = box_annotator.annotate(
        scene=frame,
        detections=detections,
        labels=labels
    )

    # Умеьшаем рамки для детектирования в n раз
    n = 100
    if result.boxes.id is not None:
        for i in range(len(detections.xyxy)):
            x1 = detections.xyxy[i][0]
            y1 = detections.xyxy[i][1]
            x2 = detections.xyxy[i][2]
            y2 = detections.xyxy[i][3]
            detections.xyxy[i][0] = x1 + (x2 - x1) / 2 - (x2 - x1) / n
            detections.xyxy[i][1] = y1 + (y2 - y1) / 2 - (y2 - y1) / n
            detections.xyxy[i][2] = x1 + (x2 - x1) / 2 + (x2 - x1) / n
            detections.xyxy[i][3] = y1 + (y2 - y1) / 2 + (y2 - y1) / n

    # Calculate
    for np1 in range(polygons):
        labels_obj[np1].pop(0)  # Удаляем верхний слой(самый старый)
        labels_obj[np1].append([])
    if detections.tracker_id is not None:  # если объекты обнаружены
        for pl in range(polygons):  # добавляем метки в полигоны
            cnt = len(detections.xyxy)  # сколько объектов найдено на кадре
            for pers in range(cnt):
                x = detections.xyxy[pers][0]
                y = detections.xyxy[pers][1]
                p = detections.tracker_id[pers]
                if inPolygon(x, y, POLYGON_COORDS[pl]):
                    # print(p, x, y, 'In Poligon ',pl)
                    r = inLabel(p, labels_obj, pl)
                    if r:  # был ли объект в другом полигоне, если да, то удаляем его и прибавляем счетчик
                        delFromLabel(p, labels_obj)
                        persons[pl] += 1
                        lf.write(str(datetime.now()) + (';In; 1' if pl else ';Out;1') + '\n')
                        print(p, end=' ')
                    else:
                        labels_obj[pl][deep - 1].append(p)  # добавляем метку объекта в полигон
    print('Out',flatten_and_sort(labels_obj[0]),' In',flatten_and_sort(labels_obj[1]), persons)
        # print_labels_obj('Kadr '+str(kadr))

    # text
    clr = sv.Color.red()
    frame = sv.draw.utils.draw_text(scene=frame, text='Out:' + str(persons[0]), text_anchor=sv.Point(x=150, y=150),
                                    text_color=clr, text_scale=1)
    frame = sv.draw.utils.draw_text(scene=frame, text=' In:' + str(persons[1]), text_anchor=sv.Point(x=150, y=200),
                                    text_color=clr, text_scale=1)
    if edit:
        frame = sv.draw.utils.draw_text(scene=frame, text=' Deep:' + str(deep),
                                        text_anchor=sv.Point(x=int(frame.shape[1] * 0.9), y=int(frame.shape[0] * 0.05)),
                                        text_color=clr, text_scale=1)
    if save_key:
        frame = sv.draw.utils.draw_text(scene=frame, text='Save picture',
                                        text_anchor=sv.Point(x=int(frame.shape[1] * 0.9), y=int(frame.shape[0] * 0.1)),
                                        text_color=clr, text_scale=1)
    if save_video_key:
        frame = sv.draw.utils.draw_text(scene=frame, text='Record mode',
                                        text_anchor=sv.Point(x=int(frame.shape[1] * 0.9), y=int(frame.shape[0] * 0.15)),
                                        text_color=clr, text_scale=1)

    # Рисуем прямоугольные рамки центров объектов на кадре
    frame = box_annotator_c.annotate(
        scene=frame,
        detections=detections,
        labels=None, skip_label=True
    )

    # Рисуем полигон на кадре
    if lbd:
        POLYGON_COORDS[coord[0]][coord[1]] = [mx, my]
    ee = POLYGON_COORDS[0]
    ww = np.array(ee)
    frame = cv2.polylines(frame, [ww], isClosed, color, thickness)
    frame = cv2.polylines(frame, [np.array(POLYGON_COORDS[1])], isClosed, color, thickness)
    if lbd:
        cv2.circle(frame, (mx, my), radius, color, -1)
    # Custom window
    cv2.namedWindow('camera', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('camera', frame)
    cv2.resizeWindow('camera', 1280, 724)
    # save video
    if save_video_key:
        # fv_ish.write_frame(frame=img)
        fv_end.write_frame(frame=frame)

    if edit: cv2.setMouseCallback('camera', mouse_events)
    wk = 30
    # Выходим из цикла, если пользователь нажал клавишу ESC
    key = cv2.waitKey(wk)
    if (key == 27):
        break
    if key == ord('r'):  # Запись видео
        save_video_key = not save_video_key
        if save_video_key:
            file_out = str(datetime.now().date()) + '_' + str(datetime.now().hour) + '-' + str(
                datetime.now().minute) + '.mp4'
            ##fv_ish=sv.VideoSink(target_path=vid_path + 'orig' + file_out, video_info=sv.VideoInfo(width=frame.shape[1], height=frame.shape[0], fps=25))
            fv_end = sv.VideoSink(target_path=vid_path + 'proc' + file_out,
                                  video_info=sv.VideoInfo(width=frame.shape[1], height=frame.shape[0], fps=25))
            # fv_ish.__enter__()
            fv_end.__enter__()
        else:
            # fv_ish.__exit__(0,0,0)
            fv_end.__exit__(0, 0, 0)
    if key == ord('s'):  # Запись кадра
        save_key = not save_key
    if key == ord('p'):  # Запись полигонов
        update_setting(ini_file, 'camera' + str(camera), 'Out',
                       ';'.join([','.join(map(str, sublist)) for sublist in POLYGON_COORDS[0]]))
        update_setting(ini_file, 'camera' + str(camera), 'In',
                       ';'.join([','.join(map(str, sublist)) for sublist in POLYGON_COORDS[1]]))
        update_setting(ini_file, 'camera' + str(camera), 'deep', str(deep))

    if key == ord('e'):  # Вкл/выкл режим редактирования полигонов
        edit = not edit
    if edit and key == ord('q'):
        deep += 1
        labels_obj.append([])
    if edit and key == ord('a'):
        deep -= 1
        labels_obj.pop(0)  # Удаляем верхний слой(самый старый)
    if edit and key == ord(' '):
        if wk == 0:
            wk = 30
        else:
            wk = 0  # pause
            print('Pause', wk)
    if save_key and (kadr - saved_kadr) > 10 and len(dt) > 2:
        saved_kadr = kadr
        cv2.imwrite(img_path + str(camera * 100000 + kadr) + '.jpg', img)
        with open(txt_path + str(camera * 100000 + kadr) + '.txt', "w+") as l:
            cnt = len(dt)  # сколько объектов найдено на кадре
            for i in range(cnt):
                w = (dt[i][2] - dt[i][0]) / frame.shape[1]
                h = (dt[i][3] - dt[i][1]) / frame.shape[0]
                x = str(dt[i][0] / frame.shape[1] + w / 2)
                y = str(dt[i][1] / frame.shape[0] + h / 2)
                st = '0 ' + x + ' ' + y + ' ' + str(w) + ' ' + str(h)
                l.write(st + '\n')
        print('Saved', str(camera * 10000 + kadr))
cv2.destroyAllWindows()
lf.close()
if save_video_key:
    # fv_ish.__exit__(0,0,0)
    fv_end.__exit__(0, 0, 0)
