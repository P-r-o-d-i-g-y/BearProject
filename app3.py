import streamlit as st
import cv2
import pandas as pd
from ultralytics import YOLO
import tempfile
from datetime import timedelta, datetime
import json
import os
from io import BytesIO

# --- НАСТРОЙКИ СТРАНИЦЫ ---
st.set_page_config(page_title="Bear Detector", layout="wide")

@st.cache_resource
def load_model():
    return YOLO('yolov8s.pt')

model = load_model()

# --- ФУНКЦИИ ---
def save_to_history(filename, total_bears, duration_sec):
    history = []
    if os.path.exists("detection_history.json"):
        with open("detection_history.json", 'r', encoding='utf-8') as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []
    
    new_record = {
        "дата_время": datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
        "источник": filename,
        "медведей_найдено": total_bears,
        "длительность_сек": duration_sec
    }
    history.append(new_record)
    
    with open("detection_history.json", 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def generate_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Обнаружения')
        worksheet = writer.sheets['Обнаружения']
        for column in df:
            column_width = max(df[column].astype(str).map(len).max(), len(column)) + 2
            col_idx = df.columns.get_loc(column)
            worksheet.column_dimensions[chr(65 + col_idx)].width = column_width
    output.seek(0)
    return output.getvalue()

# --- ИНИЦИАЛИЗАЦИЯ STATE ---
if 'history_data' not in st.session_state:
    st.session_state.history_data = []
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0
if 'processing_active' not in st.session_state:
    st.session_state.processing_active = False

# --- ИНТЕРФЕЙС ---
st.title("Детектор медведей")
st.sidebar.header("Настройки")
source_option = st.sidebar.selectbox("Источник видео", ("Загрузка видео", "Веб-камера"))

# Сброс при смене режима
if 'last_source' not in st.session_state:
    st.session_state.last_source = source_option
if st.session_state.last_source != source_option:
    st.session_state.history_data = []
    st.session_state.processing_active = False
    st.session_state.last_source = source_option

cap = None
temp_path = None
is_camera = False

# Логика выбора источника
if source_option == "Загрузка видео":
    # ВОТ ОНА, ВАЖНАЯ СТРОЧКА: key обновляется, позволяя загружать то же видео заново
    uploaded_file = st.file_uploader(
        "Загрузите видео", 
        type=["mp4", "avi"], 
        key=f"uploader_{st.session_state.uploader_key}"
    )

    if uploaded_file is not None:
        # Кнопка старта, чтобы не запускать обработку сразу при выборе файла (опционально)
        if st.button("Старт обработки") or st.session_state.processing_active:
             # Сохраняем файл во временную папку
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
            tfile.write(uploaded_file.getvalue())
            tfile.flush() # Важно сбросить буфер на диск
            temp_path = tfile.name
            
            cap = cv2.VideoCapture(temp_path)
            st.session_state.filename = uploaded_file.name
            st.session_state.processing_active = True
            is_camera = False

elif source_option == "Веб-камера":
    if st.button("Включить камеру") or st.session_state.processing_active:
        cap = cv2.VideoCapture(0)
        st.session_state.filename = "Веб-камера"
        st.session_state.processing_active = True
        is_camera = True

# --- ОБРАБОТКА ВИДЕО ---
# Мы входим сюда только если cap создан (то есть нажали кнопку или процесс идет)
if cap is not None and cap.isOpened():
    
    st_frame = st.empty()  # Плейсхолдер для видео
    stop_button = st.button("Остановить") # Кнопка стоп рисуется ДО цикла

    if stop_button:
        # Если нажали стоп - просто прерываем, Streamlit сам перезагрузит скрипт
        st.session_state.processing_active = False
        cap.release()
        st.rerun()

    st.session_state.history_data = [] # Очищаем старую статистику перед новым прогоном
    
    frame_count = 0
    last_logged_sec = -1
    start_time_proc = datetime.now()

    # --- ГЛАВНЫЙ ЦИКЛ (Как в старом коде - БЫСТРЫЙ) ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # Видео закончилось
            break

        frame_count += 1
        
        # Модель (каждый 5-й кадр для скорости)
        if frame_count % 5 == 0:
            results = model.predict(frame, conf=0.4, classes=[21], iou=0.5, verbose=False)
            
            # Статистика
            bears_count = len(results[0].boxes)
            if bears_count > 0:
                if is_camera:
                    current_sec = int(datetime.now().timestamp())
                    video_time = datetime.now().strftime("%H:%M:%S")
                else:
                    msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                    current_sec = int(msec / 1000)
                    video_time = str(timedelta(milliseconds=int(msec))).split('.')[0]

                if current_sec != last_logged_sec:
                    confidences = [float(c) for c in results[0].boxes.conf]
                    avg_conf = sum(confidences) / len(confidences)
                    st.session_state.history_data.append({
                        "Время": video_time,
                        "Медведей": bears_count,
                        "Уверенность": f"{avg_conf:.2f}"
                    })
                    last_logged_sec = current_sec

            # Рисуем
            frame_with_boxes = results[0].plot(img=frame)
            st_frame.image(frame_with_boxes, channels="BGR")

    # --- ЗАВЕРШЕНИЕ ОБРАБОТКИ ---
    cap.release()
    st.session_state.processing_active = False # Снимаем флаг активности
    
    # Расчет длительности
    duration = int((datetime.now() - start_time_proc).total_seconds())
    
    # Сохраняем в историю JSON
    if len(st.session_state.history_data) > 0:
        total_bears = max(row["Медведей"] for row in st.session_state.history_data)
        save_to_history(st.session_state.filename, total_bears, duration)

    # ВАЖНО: Обновляем ключ загрузчика, чтобы поле очистилось и можно было залить то же видео
    if not is_camera:
        st.session_state.uploader_key += 1
    
    st.success("Обработка завершена!")
    st.rerun() # Перезагружаем страницу, чтобы обновить интерфейс и показать таблицу

# --- ВЫВОД РЕЗУЛЬТАТОВ (ПОСЛЕ ЦИКЛА) ---
if len(st.session_state.history_data) > 0 and not st.session_state.processing_active:
    st.write("Результаты последнего анализа:")
    df = pd.DataFrame(st.session_state.history_data)
    st.dataframe(df)
    
    excel_data = generate_excel(df)
    st.download_button(
        label="Скачать отчет (Excel)",
        data=excel_data,
        file_name="report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# --- БОКОВАЯ ПАНЕЛЬ ИСТОРИИ ---
st.sidebar.divider()
st.sidebar.header("История запросов")
if os.path.exists("detection_history.json"):
    with open("detection_history.json", 'r', encoding='utf-8') as f:
        try:
            history = json.load(f)
            if history:
                history_df = pd.DataFrame(history)
                st.sidebar.dataframe(history_df, use_container_width=True)
                st.sidebar.metric("Всего анализов", len(history))
        except:
            st.sidebar.info("Ошибка чтения истории")