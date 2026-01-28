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

# если режим изменился очищаем старую статистику
def reset_run_state():
    if st.session_state.active_cap is not None:
        st.session_state.active_cap.release()
        st.session_state.active_cap = None
    st.session_state.history_data = []
    st.session_state.stopped = False
    st.session_state.processing = False
    st.session_state.saved_to_history = False
    st.session_state.video_duration = 0


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
        st.session_state.filename = uploaded_file.name

        video_bytes = uploaded_file.getvalue()  # не "съедает" буфер как read()
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
        tfile.write(video_bytes)
        tfile.flush()

        cap = cv2.VideoCapture(tfile.name)
        is_camera = False
        st.session_state.processing = True
        st.session_state.start_time = datetime.now()
        st.session_state.is_camera_mode = False

elif source_option == "Веб-камера":
    # если выбрали камеру, создаем кнопку запуска
    if st.button("Включить камеру"):
        reset_run_state()  #чтобы очистить старые данные
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

        # сохранить в JSON
        if len(st.session_state.history_data) > 0 and not st.session_state.saved_to_history:
            total_bears = max(row["Медведей"] for row in st.session_state.history_data)
            save_to_history(st.session_state.filename, total_bears, st.session_state.video_duration)
            st.session_state.saved_to_history = True

        cap.release()
        st.session_state.active_cap = None
        st.session_state.processing = False
        st.session_state.stop_requested = False  # сброс флага

        # автосброс file_uploader
        if source_option == "Загрузка видео" and not is_camera:
            st.session_state.uploader_key += 1
            st.rerun()
   
    else:  #чтобы он не выполнялся при остановке
        # счетчик кадров
        frame_count = 0 
        last_logged_sec = -1                                       # чтобы не спамить одну секунду
        start_time = datetime.now()  # Для подсчета длительности

        #цикл модели-----------------------------------------
        while cap.isOpened() and not stop_button:
            
            ret, frame = cap.read()
            if not ret:
                st.write("Видео закончилось")
                st.session_state.stopped = False
                break
            frame_count += 1
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
            if frame_count % 5 == 0:
                
                results = model.predict(
                    frame, 
                    conf=0.4,                               #уверенность
                    classes=[21],
                    iou=0.5,                                #убирает дубли
                    verbose=False                           #False чтобы терминал не спамил
                )

                # статистика
                bears_count = len(results[0].boxes)
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
                st_frame.image(frame_with_boxes, channels="BGR")

        st.session_state.video_duration = int((datetime.now() - st.session_state.start_time).total_seconds())        
   

        cap.release()                                       #освобождение ресурсов
        st.session_state.active_cap = None
        st.session_state.processing = False
        
        # сохранение в   json
        if len(st.session_state.history_data) > 0 and not st.session_state.saved_to_history:
            total_bears = max(row["Медведей"] for row in st.session_state.history_data)
            save_to_history(st.session_state.filename, total_bears, st.session_state.video_duration)
            st.session_state.saved_to_history = True

        # автосброс file_uploader и перезапуск
        if source_option == "Загрузка видео" and not is_camera:
            st.session_state.uploader_key += 1
            st.rerun()
    
# вывод отчета
if len(st.session_state.history_data) > 0:

    if st.session_state.stopped:
        st.info("Обработка остановлена. Текущий результат:")
    else:
        st.success("Обработка завершена. Итоговый результат:")

    df = pd.DataFrame(st.session_state.history_data)
    st.dataframe(df)                                       # показать таблицу

    #отчет статистики 
    excel_data = generate_excel(df)
    st.download_button(
        label="Скачать отчет (Excel)",
        data=excel_data,
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

