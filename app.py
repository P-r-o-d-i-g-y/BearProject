import streamlit as st                                    #заменяет htlml css js
import cv2                                                #достает картинки для ии (opencv)
import pandas as pd                                       #нужно для Excel
from ultralytics import YOLO                              #предобученная модель 
import tempfile                                           #тащит видео из рам на диск для opencv
from datetime import datetime                             #для времени обнаружения

#настройки-------------------------
CONFIDENCE_THRESHOLD = 0.5

@st.cache_resource                                          #чтобы модель не загружалась заново и осталась в памяти
# загрузка модели с предобучением
def load_model():
    model = YOLO('yolov8n.pt')                              
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
    #цикл модели-----------------------------------------
    while cap.isOpened() and not stop_button:
        
        ret, frame = cap.read()
        if not ret:
            st.write("Видео закончилось")
            break
        frame_count += 1

        if frame_count % 10 != 0:
             st_frame.image(frame, channels="BGR")
             continue

        results = model.predict(frame, conf=0.4, classes=[21])    #в базе COCO 21- Медведь
        frame_with_boxes = results[0].plot()                      #квадраты на кадре
        
        # СБОР СТАТИСТИКИ 
        bears_count = len(results[0].boxes) # Сколько рамок нашел
        if bears_count > 0:
            history_data.append({
                "Время": datetime.now().strftime("%H:%M:%S"),
                "Медведей": bears_count
            })

        st_frame.image(frame_with_boxes, channels="BGR")
    cap.release()                                                 #освобождение ресурсов

    # --- ВЫВОД ОТЧЕТА (добавлено) ---
    if len(history_data) > 0:
        st.success("Обработка завершена.")
        df = pd.DataFrame(history_data)
        st.dataframe(df) # Показать таблицу
        
        # Скачать CSV
        csv = df.to_csv().encode('utf-8')
        st.download_button("Скачать отчет", csv, "report.csv")
