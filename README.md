# Bear Detection AI Web App

A web-based AI application designed to detect bears near populated areas. Built with Python and Streamlit, this project utilizes the YOLOv8s (Small) object detection model to process video files and live webcam streams, providing real-time bounding boxes and confidence metrics.

## Key Features
* **Dual Input Modes:** Supports video file uploads (`.mp4`, `.avi`) and real-time live webcam processing
* **Optimized Object Detection:** Powered by `ultralytics` YOLOv8s, achieving an optimal balance between processing speed and detection accuracy. To maintain a smooth UI experience and performance, the application processes every 5th frame of the video stream.
* **Detection Analytics:** Tracks the maximum number of bears detected simultaneously and calculates average confidence scores across the video timeline.
* **History Logging:** Automatically saves detection sessions (including date, source name, total bears found, and processing duration) into a local `detection_history.json` file.
* **Exportable Reports:** Generates downloadable Excel (`.xlsx`) reports using `pandas` and `openpyxl`, containing detailed timestamped detection statistics for further analysis.

## How to Build and Run
To run this project, you need Python installed on your machine.

**1. Clone this repository:**
`git clone https://github.com/P-r-o-d-i-g-y/BearProject.git`

**2. Navigate to the project directory:**
`cd BearProject`

**3. Install required dependencies:**
Make sure you have the necessary Python libraries installed. You can install them via pip:
`pip install streamlit opencv-python pandas ultralytics openpyxl`

**4. Run the application:**
Start the Streamlit server using the following command:
`python -m streamlit run app.py`
