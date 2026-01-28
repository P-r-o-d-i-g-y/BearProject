import streamlit as st                                    #заменяет htlml css js
import cv2                                                #достает картинки для ии (opencv)
import pandas as pd                                       #нужно для Excel
from ultralytics import YOLO                              #предобученная модель 
import tempfile                                           #тащит видео из рам на диск для opencv
from datetime import timedelta, datetime                  #для времени обнаружения
import json                                               #история запросов
import os
from io import BytesIO

# python -m streamlit run app.py

@st.cache_resource                                          #чтобы модель не загружалась заново и осталась в памяти
# загрузка модели с предобучением
def load_model():
    model = YOLO('yolov8s.pt')                              
    return model

model = load_model()

# ф-ия сохранения в историю запросов json
def save_to_history(filename, total_bears, duration_sec):
    history = []
    
    if os.path.exists("detection_history.json"):
        with open("detection_history.json", 'r', encoding='utf-8') as f:
            history = json.load(f)
    
    new_record = {
        "дата_время": datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
        "источник": filename,
        "медведей_найдено": total_bears,
        "длительность_сек": duration_sec
    }
    history.append(new_record)
    
    with open("detection_history.json", 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

#генерация отчета exel
def generate_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Обнаружения')
        
        # Автоматическая ширина колонок
        worksheet = writer.sheets['Обнаружения']
        for column in df:
            column_width = max(df[column].astype(str).map(len).max(), len(column)) + 2
            col_idx = df.columns.get_loc(column)
            worksheet.column_dimensions[chr(65 + col_idx)].width = column_width
    
    output.seek(0)
    return output.getvalue()


#интерфейс--------------------------
st.title("Детектор медведей вблизи населенных пунктов")
st.sidebar.header("Настройки")
source_option = st.sidebar.selectbox("Источник видео", ("Загрузка видео", "Веб-камера"))
result_placeholder = st.empty() #контейнер результата

# отслеживание смены режима
if 'last_source' not in st.session_state:
    st.session_state.last_source = source_option

if 'history_data' not in st.session_state:
    st.session_state.history_data = []

if 'processing' not in st.session_state:
    st.session_state.processing = False

if 'stopped' not in st.session_state:
    st.session_state.stopped = False

if 'active_cap' not in st.session_state:  
    st.session_state.active_cap = None    

if 'filename' not in st.session_state:
    st.session_state.filename = ""

if 'video_duration' not in st.session_state:
    st.session_state.video_duration = 0

if 'saved_to_history' not in st.session_state:
    st.session_state.saved_to_history = False

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if "stop_requested" not in st.session_state:
    st.session_state.stop_requested = False

if "start_time" not in st.session_state:
    st.session_state.start_time = None

if "is_camera_mode" not in st.session_state:  
    st.session_state.is_camera_mode = False

if "excel_data" not in st.session_state:
    st.session_state.excel_data = None

if "excel_key" not in st.session_state:
    st.session_state.excel_key = None

if "max_bears_seen" not in st.session_state:
    st.session_state.max_bears_seen = 0

if "show_result" not in st.session_state:
    st.session_state.show_result = False

# если режим изменился очищаем старую статистику
def reset_run_state():
    if st.session_state.active_cap is not None:
        st.session_state.active_cap.release()
        st.session_state.active_cap = None
    
    # удалить предыдущий temp-файл, если был
    temp_path = st.session_state.get("temp_video_path")
    if temp_path and os.path.exists(temp_path):
        os.remove(temp_path)
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
    st.session_state.show_result = False

def save_uploaded_to_temp(uploaded_file):
            suffix = os.path.splitext(uploaded_file.name)[1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tfile:
                # getbuffer() — без лишней копии huge-bytes в переменную
                tfile.write(uploaded_file.getbuffer())
                return tfile.name

# если режим изменился — очищаем старую статистику
if st.session_state.last_source != source_option:
    reset_run_state()
    st.session_state.last_source = source_option
    st.rerun()

cap = None
is_camera = False

if source_option == "Загрузка видео":

    uploaded_file = st.file_uploader(
    "Загрузите видео",
    type=["mp4", "avi"],
    key=f"video_uploader_{st.session_state.uploader_key}",
)

    # запуск обработки (лучше через явную кнопку, чтобы не стартовало "само" на каждом rerun)
    if uploaded_file is not None and st.button("Старт обработки"):
        reset_run_state()
        result_placeholder.empty()
        st.session_state.filename = uploaded_file.name

        temp_path = save_uploaded_to_temp(uploaded_file)
        st.session_state.temp_video_path = temp_path  # чтобы потом удалить
        cap = cv2.VideoCapture(temp_path)
        
        is_camera = False
        st.session_state.processing = True
        st.session_state.start_time = datetime.now()
        st.session_state.is_camera_mode = False

elif source_option == "Веб-камера":
    # если выбрали камеру, создаем кнопку запуска
    if st.button("Включить камеру"):
        reset_run_state()  #чтобы очистить старые данные
        result_placeholder.empty() #убрать кнопку скачать
        st.session_state.filename = "Веб-камера"
        cap = cv2.VideoCapture(0)                               # 0 это вебкамера
        is_camera = True
        st.session_state.processing = True
        # st.session_state.history_data = []                      #очищаем при новом запуске
        st.session_state.stopped = False
        st.session_state.start_time = datetime.now()
        st.session_state.is_camera_mode = True
# восстановить cap после rerun (например, при нажатии "Остановить")
if cap is None and st.session_state.processing and st.session_state.active_cap is not None:
    cap = st.session_state.active_cap
    is_camera = st.session_state.is_camera_mode

#проверка на файл
if cap is not None:
    
    st.session_state.active_cap = cap

    st_frame = st.empty()                                       #элемент на странице, куда будем выводить кадры
    stop_button = st.button("Остановить")

    if stop_button:
        st.session_state.stop_requested = True
    
    # Сброс при новом файле
    if not st.session_state.processing:
        st.session_state.history_data = []
        st.session_state.processing = True
        st.session_state.stopped = False


        #обработка остановки ДО цикла
    if st.session_state.stop_requested:
        st.session_state.stopped = True
        st_frame.empty()

        if st.session_state.start_time is not None:
            st.session_state.video_duration = int((datetime.now() - st.session_state.start_time).total_seconds())

        # сохраниТЬ в JSON
        if not st.session_state.saved_to_history:
            total_bears = st.session_state.max_bears_seen  # будет 0 если не нашли
            save_to_history(st.session_state.filename, total_bears, st.session_state.video_duration)
            st.session_state.saved_to_history = True

        cap.release()
        # удалить временный файл (только для режима "Загрузка видео")
        temp_path = st.session_state.get("temp_video_path")
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        st.session_state.temp_video_path = None
        st.session_state.active_cap = None
        st.session_state.processing = False
        st.session_state.stop_requested = False  # сброс флага

        # автосброс file_uploader только для режима "Загрузка видео"
        if source_option == "Загрузка видео" and not is_camera:
            st.session_state.uploader_key += 1

        st.session_state.show_result = True
        # st.session_state.stopped = False
        # rerun всегда, чтобы UI обновился и кнопка исчезла сразу
        st.rerun()
   
    else:  #чтобы он не выполнялся при остановке
        # счетчик кадров
        frame_count = 0 
        last_logged_sec = -1                                       # чтобы не спамить одну секунду
        start_time = datetime.now()  # Для подсчета длительности

        #цикл модели-----------------------------------------
        while cap.isOpened() and not stop_button:
            
            ret = cap.grab()
            if not ret:
                st.write("Видео закончилось")
                break
            
            frame_count += 1
            if frame_count % 5 != 0:
                continue  # кадр пропускаем БЕЗ декодирования

            ret, frame = cap.retrieve()
            if not ret:
                break
            #тамкод----------------------------------------------
            #разная логика для времени реал тайм/видео
            if is_camera:
                video_time = datetime.now().strftime("%H:%M:%S")
                current_sec = int(datetime.now().timestamp())  #метка для каждой секунды
            else:
                msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                video_time = str(timedelta(milliseconds=int(msec))).split('.')[0]
                current_sec = int(msec / 1000)

            #модель---------------------------------------------
            
                
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
                if current_sec != last_logged_sec:
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
            # уменьшение кадра перед отправкой в браузер (ускоряет Streamlit)
            max_w = 960
            h, w = frame_with_boxes.shape[:2]
            if w > max_w:
                scale = max_w / w
                frame_with_boxes = cv2.resize(frame_with_boxes, (max_w, int(h * scale)))
                
            st_frame.image(frame_with_boxes, channels="BGR")

        st.session_state.video_duration = int((datetime.now() - st.session_state.start_time).total_seconds())        
   

        cap.release()                                       #освобождение ресурсов
        
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

        # автосброс file_uploader и перезапуск
        if source_option == "Загрузка видео" and not is_camera:
            st.session_state.uploader_key += 1
            
        st.rerun()
    
# вывод отчета
with result_placeholder.container():
    if (not st.session_state.processing) and st.session_state.show_result:

        if st.session_state.stopped:
            st.info("Обработка остановлена. Текущий результат:")
        else:
            st.success("Обработка завершена. Итоговый результат:")

        if st.session_state.max_bears_seen == 0:
            st.warning("Медведей не найдено (0).")
        else:
            df = pd.DataFrame(st.session_state.history_data)
            st.dataframe(df)                                        # показать таблицу


            # Excel генерим только если данные изменились
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

