import streamlit as st                                    #заменяет htlml css js
import cv2                                                #достает картинки для ии (opencv)
import pandas as pd                                       #нужно для Excel
from ultralytics import YOLO                              #предобученная модель 
import tempfile                                           #тащит видео из рам на диск для opencv
from datetime import timedelta, datetime                  #для времени обнаружения

# python -m streamlit run app.py

@st.cache_resource                                          #чтобы модель не загружалась заново и осталась в памяти
# загрузка модели с предобучением
def load_model():
    model = YOLO('yolov8s.pt')                              
    return model

model = load_model()

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

# если режим изменился очищаем старую статистику
if st.session_state.last_source != source_option:
    if st.session_state.active_cap is not None:
        st.session_state.active_cap.release()                     #освобождаем камеру
        st.session_state.active_cap = None
    st.session_state.history_data = []
    st.session_state.stopped = False
    st.session_state.processing = False
    st.session_state.last_source = source_option

cap = None                                                       #переменная файл
is_camera = False

#кнопка загрузки файла
if source_option == "Загрузка видео":
    uploaded_file = st.file_uploader("Загрузите видео", type=['mp4', 'avi'])
    if uploaded_file is not None:
        #сохранение файл во временную папку, потому что OpenCV не сможет его открыть
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        is_camera = False

elif source_option == "Веб-камера":
    # если выбрали камеру, создаем кнопку запуска
    if st.button("Включить камеру"):
        cap = cv2.VideoCapture(0)                               # 0 это вебкамера
        is_camera = True
        st.session_state.processing = True
        st.session_state.history_data = []                      #очищаем при новом запуске
        st.session_state.stopped = False


#проверка на файл
if cap is not None:
    
    st.session_state.active_cap = cap

    st_frame = st.empty()                                       #элемент на странице, куда будем выводить кадры
    stop_button = st.button("Остановить")
    
    # Сброс при новом файле
    if not st.session_state.processing:
        st.session_state.history_data = []
        st.session_state.processing = True
        st.session_state.stopped = False

    # счетчик кадров
    frame_count = 0 
    last_logged_sec = -1                                       # чтобы не спамить одну секунду

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
                    st.session_state.history_data.append({
                        "Время": video_time,  
                        "Медведей": bears_count
                    })
                    last_logged_sec = current_sec
            #рамка
            frame_with_boxes = results[0].plot(img=frame)
            st_frame.image(frame_with_boxes, channels="BGR")

    if stop_button:
        was_stopped = True
        st_frame.empty()                                #убираем видео с экрана
   
    cap.release()                                       #освобождение ресурсов
    st.session_state.active_cap = None
    st.session_state.processing = False
    

    # вывод отчета
if len(st.session_state.history_data) > 0:

    if st.session_state.stopped:
        st.info("Обработка остановлена. Текущий результат:")
    else:
        st.success("Обработка завершена. Итоговый результат:")

    df = pd.DataFrame(st.session_state.history_data)
    st.dataframe(df)                                       # показать таблицу
        
    # скачать CSV
    csv = df.to_csv().encode('utf-8')
    st.download_button("Скачать отчет", csv, "report.csv")
