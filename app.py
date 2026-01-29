import streamlit as st                                    #заменяет htlml css js
import cv2                                                #достает картинки для ии (opencv)
import pandas as pd                                       #нужно для Excel
from ultralytics import YOLO                              #предобученная модель 
import tempfile                                           #тащит видео из рам на диск для opencv
from datetime import timedelta, datetime                  #для времени обнаружения
import json                                               #история запросов
import os                                                 #для взаимодействия с операционной системой и файловой системой
from io import BytesIO                                    #создает файлоподобный объект в оперативной памяти для работы с байтовыми данными

# python -m streamlit run app.py

@st.cache_resource                                        #чтобы модель не загружалась заново и осталась в памяти
# загрузка модели с предобучением
def load_model():
    model = YOLO('yolov8s.pt')                              
    return model
model = load_model()

#интерфейс веб 
st.title("Детектор медведей вблизи населенных пунктов")
browse_slot = st.empty()                                  #выше result_placeholder чтобы не съезжал
st.sidebar.header("Настройки")
source_option = st.sidebar.selectbox("Источник видео", ("Загрузка видео", "Веб-камера"))
result_placeholder = st.container() #контейнер результата

#инициализация переменных состояния между rerun
if 'last_source' not in st.session_state:                  # отслеживание смены режима
    st.session_state.last_source = source_option

if 'history_data' not in st.session_state:                 #для создания DataFrame в экселе
    st.session_state.history_data = []

if 'processing' not in st.session_state:                   #для блокировки кнопки старта во время работы
    st.session_state.processing = False

if 'stopped' not in st.session_state:                      #была ли обработка остановленна пользователем
    st.session_state.stopped = False

if 'active_cap' not in st.session_state:                   #сохранения соединения с камерой/видео между rerun
    st.session_state.active_cap = None    

if 'filename' not in st.session_state:                     #название файла для истории запросов
    st.session_state.filename = ""

if 'video_duration' not in st.session_state:               
    st.session_state.video_duration = 0

if 'saved_to_history' not in st.session_state:             #предотвращает повторное сохранение результатов
    st.session_state.saved_to_history = False

if "uploader_key" not in st.session_state:                 #для повторного загруза видео
    st.session_state.uploader_key = 0

if "stop_requested" not in st.session_state:               #для сохранения данных если остановили
    st.session_state.stop_requested = False

if "start_time" not in st.session_state:                   #время начала обработки
    st.session_state.start_time = None

if "is_camera_mode" not in st.session_state:               #True - веб-камера, False - видеофайл
    st.session_state.is_camera_mode = False

if "excel_data" not in st.session_state:                   #чтобы не генерировать файл заново в rerun
    st.session_state.excel_data = None

if "excel_key" not in st.session_state:                    #длина списка history_data, отслеживает изменение данных
    st.session_state.excel_key = None

if "max_bears_seen" not in st.session_state:               
    st.session_state.max_bears_seen = 0

if "show_result" not in st.session_state:                  #чтобы кнопка скачать была после обработки
    st.session_state.show_result = False

# функции
# ф-ия сохранения в историю запросов json
def save_to_history(filename, total_bears, duration_sec):
    history = []
    
    # открытие и запись / создание
    if os.path.exists("detection_history.json"):
        with open("detection_history.json", 'r', encoding='utf-8') as f:
            history = json.load(f) #json в pyсписок
    
    new_record = {
        "дата_время": datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
        "источник": filename,
        "медведей_найдено": total_bears,
        "длительность_сек": duration_sec
    }
    history.append(new_record)
    # сохранение
    with open("detection_history.json", 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

# генерация отчета exel
def generate_excel(df):
    output = BytesIO()                                                    #создание объекта в рам
    with pd.ExcelWriter(output, engine='openpyxl') as writer:             #для записи
        df.to_excel(writer, index=False, sheet_name='Обнаружения')        
        
        # автоматическая ширина колонок
        worksheet = writer.sheets['Обнаружения']
        for column in df:
            column_width = max(df[column].astype(str).map(len).max(), len(column)) + 2
            col_idx = df.columns.get_loc(column)
            worksheet.column_dimensions[chr(65 + col_idx)].width = column_width
    
    output.seek(0) #возврат в начало буфера
    return output.getvalue()

# сброс/перезагрузка
def reset_run_state():
    if st.session_state.active_cap is not None: #проверка тек обработки
        st.session_state.active_cap.release()   #закрывает
        st.session_state.active_cap = None      #удаление ссылки на камеру/видео
    # удалить предыдущий temp-файл
    temp_path = st.session_state.get("temp_video_path")
    if temp_path and os.path.exists(temp_path):
        os.remove(temp_path)                    #освобождение
    #сбросы
    st.session_state.temp_video_path = None     
    st.session_state.history_data = []          
    st.session_state.max_bears_seen = 0
    st.session_state.stopped = False
    st.session_state.processing = False
    st.session_state.saved_to_history = False
    st.session_state.video_duration = 0
    st.session_state.excel_key = None
    st.session_state.excel_data = None
    st.session_state.start_time = None
    st.session_state.stop_requested = False
    # убрать старый результат из UI
    result_placeholder.empty()
    st.session_state.show_result = False

# ф-ия из рам в диск, чтобы OpenCV его читал
def save_uploaded_to_temp(uploaded_file):
            suffix = os.path.splitext(uploaded_file.name)[1].lower()   #получение разшерения
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tfile: #создать файл на диске
                # кеширует в файл на диске
                tfile.write(uploaded_file.getbuffer())
                return tfile.name

# очищение статистики при смене режима
if st.session_state.last_source != source_option:
    reset_run_state()
    st.session_state.last_source = source_option
    st.rerun() #перересовка интерфейса

#переменные
cap = None
is_camera = False
uploaded_file = None 

#блок видео
if source_option == "Загрузка видео":
    with  browse_slot.container(): #в контейнере для browse
        uploaded_file = st.file_uploader(
            "Загрузите видео",
            type=["mp4", "avi"],
            key=f"video_uploader_{st.session_state.uploader_key}", #загрузка видео с счетчиком
        )
        start_clicked = st.button(
            "Старт обработки",
            disabled=(uploaded_file is None) or st.session_state.processing,
            key="start_processing_btn",
        )
    if start_clicked:
        result_placeholder.empty()
        reset_run_state()
        #сохранение
        st.session_state.filename = uploaded_file.name
        temp_path = save_uploaded_to_temp(uploaded_file)
        st.session_state.temp_video_path = temp_path #путь, чтобы потом его удалить
        cap = cv2.VideoCapture(temp_path) #открыть видео
        is_camera = False #нужно для вида таймкода
        st.session_state.processing = True
        st.session_state.start_time = datetime.now()
        st.session_state.is_camera_mode = False
#блок вебки
elif source_option == "Веб-камера":
    if st.button("Включить камеру"):
        result_placeholder.empty()
        reset_run_state() 
        #убирание старых данных
        result_placeholder.empty() #убрать кнопку скачать
        st.session_state.filename = "Веб-камера"
        cap = cv2.VideoCapture(0)   #0 - камера ноута
        is_camera = True
        st.session_state.processing = True
        st.session_state.stopped = False
        st.session_state.start_time = datetime.now() #запуск обработки
        st.session_state.is_camera_mode = True

# !восстановить cap после rerun (чтобы убрать камеру и удалить temp на диске, но сохранить статистику)
if cap is None and st.session_state.processing and st.session_state.active_cap is not None:
    cap = st.session_state.active_cap
    is_camera = st.session_state.is_camera_mode

#основной цикл обработки при старте
if cap is not None: #проверка видео потока
    st.session_state.active_cap = cap   #сохранение на случай остановки    
    st_frame = st.empty()   #контейнер для кадров         
    stop_button = st.button("Остановить")

    if stop_button: #флаг, потому что нельзя прервать обработку
        st.session_state.stop_requested = True
    
    # сброс при новом файле перед запуском
    if not st.session_state.processing:
        st.session_state.history_data = []
        st.session_state.processing = True
        st.session_state.stopped = False

    #обработка остановки
    if st.session_state.stop_requested:
        #все убрать, мы остановились
        st.session_state.stopped = True
        st_frame.empty()
        #подсчет времени обработки
        if st.session_state.start_time is not None:
            st.session_state.video_duration = int((datetime.now() - st.session_state.start_time).total_seconds())
        # сохраниТЬ в json
        if not st.session_state.saved_to_history:
            total_bears = st.session_state.max_bears_seen  # будет 0 если не нашли
            save_to_history(st.session_state.filename, total_bears, st.session_state.video_duration)
            st.session_state.saved_to_history = True

        cap.release() #освободить камеру
        
        # удалить временный файл
        temp_path = st.session_state.get("temp_video_path")
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        st.session_state.temp_video_path = None
        st.session_state.active_cap = None
        st.session_state.processing = False
        st.session_state.stop_requested = False

        # сброс file_uploader
        if source_option == "Загрузка видео" and not is_camera:
            st.session_state.uploader_key += 1

        st.session_state.show_result = True
        st.rerun() 
    else:  #не было остановки
        frame_count = 0 
        last_logged_sec = -1         # чтобы не спамить записи в одну милисекунду
        start_time = datetime.now()  # для подсчета длительности

        #цикл модели
        while cap.isOpened() and not stop_button:
            
            ret = cap.grab() #схватить кадр
            if not ret:
                st.write("Видео закончилось")
                break
            
            #оптимизация
            frame_count += 1
            if frame_count % 5 != 0:
                continue  # кадр пропускаем без декодирования

            ret, frame = cap.retrieve()
            if not ret:
                break
            #тамкод
            #разная логика для времени реал тайм/видео
            if is_camera:
                video_time = datetime.now().strftime("%H:%M:%S")
                current_sec = int(datetime.now().timestamp())  #метка для каждой секунды
            else:
                msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                video_time = str(timedelta(milliseconds=int(msec))).split('.')[0]
                current_sec = int(msec / 1000)

            #модель (в поиске медведя)   
            results = model.predict(
                frame, 
                conf=0.4,                               #уверенность
                classes=[21],
                iou=0.5,                                #убирает дубли
                verbose=False                           #False чтобы терминал не спамил
            )

            # статистика
            bears_count = len(results[0].boxes)
            if bears_count > st.session_state.max_bears_seen:
                st.session_state.max_bears_seen = bears_count
            
            if bears_count > 0:
                # print(f"Кадр {frame_count}: найдено {bears_count} объектов. Уверенности: {[round(float(c), 2) for c in results[0].boxes.conf]}")                
                if current_sec != last_logged_sec: #антиспам, только в сек
                    #средняя уверенность
                    confidences = [float(c) for c in results[0].boxes.conf]
                    avg_conf = sum(confidences) / len(confidences)
                    st.session_state.history_data.append({
                        "Время": video_time,  
                        "Медведей": bears_count,
                        "Уверенность": f"{avg_conf:.2f}"
                    })
                    last_logged_sec = current_sec
            #рамка
            frame_with_boxes = results[0].plot(img=frame)
            # уменьшение кадра перед отправкой в браузер (ускоряет)
            max_w = 960
            h, w = frame_with_boxes.shape[:2]
            if w > max_w:
                scale = max_w / w
                frame_with_boxes = cv2.resize(frame_with_boxes, (max_w, int(h * scale)))
            st_frame.image(frame_with_boxes, channels="BGR") #дали сжатую
        #уборщик
        st.session_state.video_duration = int((datetime.now() - st.session_state.start_time).total_seconds())        
        cap.release()     #освобождение ресурсов
        temp_path = st.session_state.get("temp_video_path")
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        st.session_state.temp_video_path = None
        st.session_state.active_cap = None
        st.session_state.processing = False
        st.session_state.show_result = True
        
        # сохранениЕ в json
        if not st.session_state.saved_to_history:
            total_bears = st.session_state.max_bears_seen
            save_to_history(st.session_state.filename, total_bears, st.session_state.video_duration)
            st.session_state.saved_to_history = True

        # сброс file_uploader
        if source_option == "Загрузка видео" and not is_camera:
            st.session_state.uploader_key += 1
            
        st.rerun()
    
# вывод отчета
with result_placeholder.container(): #контейнер для статистики
    if (not st.session_state.processing) and st.session_state.show_result:

        if st.session_state.stopped:
            st.info("Обработка остановлена. Текущий результат:")
        else:
            st.success("Обработка завершена. Итоговый результат:")

        if st.session_state.max_bears_seen == 0:
            st.warning("Медведей не найдено (0).")
        else: #в найденных покажем таблицу
            df = pd.DataFrame(st.session_state.history_data)
            st.dataframe(df)                                        # показать таблицу

        #Streamlit при нажатии любой кнопки выполняет app заново    
            # excel генерим только если данные изменились
            key = len(st.session_state.history_data)  
            if st.session_state.excel_key != key:
                st.session_state.excel_data = None
                st.session_state.excel_key = key

            if st.session_state.excel_data is None:
                with st.spinner("Готовлю Excel для скачивания..."):
                    st.session_state.excel_data = generate_excel(df)
            #кнопка
            st.download_button(
                label="Скачать отчёт (Excel)",
                data=st.session_state.excel_data,
                file_name="report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# сайдбар история запросов
st.sidebar.header("История запросов")
if os.path.exists("detection_history.json"):
    with open("detection_history.json", 'r', encoding='utf-8') as f:
        history = json.load(f)
    
    if len(history) > 0:
        history_df = pd.DataFrame(history)
        st.sidebar.dataframe(history_df, width='stretch')
        st.sidebar.metric("Всего анализов", len(history))
        successful_detections = len(history_df[history_df["медведей_найдено"] > 0])
        st.sidebar.metric("Успешных обнаружений", successful_detections)
    else:
        st.sidebar.info("История пуста")
else:
    st.sidebar.info("История пуста")

