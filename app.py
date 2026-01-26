import streamlit as st                                    #заменяет htlml css js
import cv2                                                #достает картинки для ии (opencv)
import pandas as pd                                       #нужно для Excel
from ultralytics import YOLO                              #предобученная модель 
import tempfile                                           #тащит видео из рам на диск для opencv
from datetime import datetime, timedelta                  #для времени обнаружения

#настройки-------------------------
CONFIDENCE_THRESHOLD = 0.4

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

history_data = []
cap = None                                            #переменная файл

#кнопка загрузки файла
if source_option == "Загрузка видео":
    uploaded_file = st.file_uploader("Загрузите видео", type=['mp4', 'avi'])
    if uploaded_file is not None:
        #сохранение файл во временную папку, потому что OpenCV не сможет его открыть
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)

elif source_option == "Веб-камера":
    # если выбрали камеру, создаем кнопку запуска
    if st.button("Включить камеру"):
        cap = cv2.VideoCapture(0) # 0 это веб-камера



#проверка на файл
if cap  is not None:
    
    st_frame = st.empty()                                       #элемент на странице, куда будем выводить кадры
    stop_button = st.button("Остановить")
    
    # Счетчик кадров
    frame_count = 0 
    last_result = None  # для хранения последней рамки
    last_logged_sec = -1  # чтобы не спамить одну секунду
    #цикл модели-----------------------------------------
    while cap.isOpened() and not stop_button:
        
        ret, frame = cap.read()
        if not ret:
            st.write("Видео закончилось")
            break
        frame_count += 1
        #тамкод
        msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        video_time = str(timedelta(milliseconds=int(msec))).split('.')[0]
        if frame_count % 10 != 0:
            # verbose=False чтобы терминал не спамил
            results = model.predict(
                frame, 
                conf=CONFIDENCE_THRESHOLD, 
                classes=[21],
                iou=0.5,  # убирает дубли
                verbose=False
            )
            last_result = results 
            # статистика
            bears_count = len(results[0].boxes)
            if bears_count > 0:
                print(f"Кадр {frame_count}: найдено {bears_count} объектов. Уверенности: {[round(float(c), 2) for c in results[0].boxes.conf]}")
                current_sec = int(msec / 1000)
                
                if current_sec != last_logged_sec:
                    history_data.append({
                        "Таймкод": video_time,  
                        "Медведей": bears_count
                    })
                    last_logged_sec = current_sec
        

        # рисуем всегда
        if last_result is not None:
            # старая рамка на новом кадре
            frame_with_boxes = last_result[0].plot(img=frame)
            st_frame.image(frame_with_boxes, channels="BGR")
        else:
            # если сеть еще не запускалась
            st_frame.image(frame, channels="BGR")
    cap.release()                                                 #освобождение ресурсов

    # вывод отчета
    if len(history_data) > 0:
        st.success("Обработка завершена.")
        df = pd.DataFrame(history_data)
        st.dataframe(df) # Показать таблицу
        
        # Скачать CSV
        csv = df.to_csv().encode('utf-8')
        st.download_button("Скачать отчет", csv, "report.csv")
