import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from keras.models import load_model
import pickle
from collections import deque
import os

# Initialize global variables
selected_video_path = ""
selected_image_path = ""
video_capture = None
model = None
lb = None
mean = None
Queue = None
writer = None
Width = None
Height = None
graph_accuracy_window = None
graph_loss_window = None

# Load the pre-trained model and other necessary objects
def load_model_and_objects():
    global model, lb, mean
    try:
        model_path = "E:/DeepLearning project/videoClassificationModel/videoClassificationModel"
        model = load_model(model_path)
        with open("E:/DeepLearning project/model/videoclassificationbinarizer.pickle", 'rb') as f:
            lb = pickle.load(f)
        mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
    except Exception as e:
        print("Error loading model and objects:", e)

# Function to select a video file
def select_video():
    global selected_video_path
    selected_video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
    if selected_video_path:
        process_and_display_video()

# Function to select an image file
def select_image():
    global selected_image_path
    selected_image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if selected_image_path:
        process_single_frame()

# Function to process a single image frame
def process_single_frame():
    frame = cv2.imread(selected_image_path)
    if frame is None:
        print("Unable to read the image.")
        return

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (244, 224)).astype("float32")
    frame -= mean
    preds = model.predict(np.expand_dims(frame, axis=0))[0]
    label = lb.classes_[np.argmax(preds)]
    accuracy = np.max(preds)
    show_prediction(label, accuracy)

# Function to display the prediction result in a new window
def show_prediction(label, accuracy):
    prediction_window = tk.Toplevel()
    prediction_window.title("Prediction Result")

    prediction_label = tk.Label(prediction_window, text="They are playing : " + label)
    prediction_label.pack()

    accuracy_label = tk.Label(prediction_window, text="Accuracy: " + str(round(accuracy * 100, 2)) + "%")
    accuracy_label.pack()

# Function to process the video and display the output in the UI
# Function to process the video and display the output in the UI
def process_and_display_video():
    global video_capture, writer, Width, Height, Queue
    if selected_video_path == "" or model is None or lb is None or mean is None:
        print("Please make sure the model and video are selected.")
        return

    video_capture = cv2.VideoCapture(selected_video_path)
    Queue = deque(maxlen=128)

    # Create a folder to store the frames
    frame_folder = "E:/DeepLearning project/videoClassificationModel/Frames"
    if not os.path.exists(frame_folder):
        try:
            os.makedirs(frame_folder)
        except OSError as e:
            print("Error creating frame folder:", e)
            return

    frame_count = 0
    correct_predictions = 0
    total_frames = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        if Width is None or Height is None:
            (Height, Width) = frame.shape[:2]

        output = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (244, 224)).astype("float32")
        frame = np.expand_dims(frame, axis=0)
        frame -= mean
        preds = model.predict(frame)[0]
        correct_predictions=np.max(preds)
        Queue.append(preds)
        results = np.array(Queue).mean(axis=0)
        i = np.argmax(results)
        label = lb.classes_[i]
        text = "They are playing: {}".format(label)
        cv2.putText(output, text, (45, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 0, 0), 5)

       
        accuracy = round(correct_predictions * 100, 2 )

        accuracy_text = "Accuracy: {:.2f}%".format(accuracy)

        cv2.putText(output, accuracy_text, (45, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)

        # Save frame
        frame_filename = os.path.join(frame_folder, f"frame{frame_count:04d}.png")
        try:
            cv2.imwrite(frame_filename, output)
        except Exception as e:
            print(f"Error saving frame {frame_count}: {e}")

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            output_file = "E:/DeepLearning project/output_video.avi"
            writer = cv2.VideoWriter(output_file, fourcc, 30, (Width, Height), True)

        writer.write(output)
        cv2.imshow("In Progress", output)

        frame_count += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    Queue.clear()
    print("Finalizing")
    writer.release()
    video_capture.release()
    cv2.destroyAllWindows()


# Function to show the accuracy graph
def show_graph_accuracy():
    global graph_accuracy_window
    if graph_accuracy_window is None:
        graph_accuracy_window = tk.Toplevel()
        graph_accuracy_window.title("Accuracy Graph")
        graph_accuracy_window.geometry("400x300")
    else:
        graph_accuracy_window.deiconify()
    
    # Load the graph image
    graph_image_path = r"E:\Developement\DeepLearning project\videoClassificationModel\model_accuracy.png"  
    graph_image = Image.open(graph_image_path)
    graph_image = graph_image.resize((400, 300))

    # Convert the PIL image to a Tkinter-compatible format
    graph_image_tk = ImageTk.PhotoImage(graph_image)

    # Display the graph image in the window
    graph_label = tk.Label(graph_accuracy_window, image=graph_image_tk)
    graph_label.image = graph_image_tk  # Keep a reference to prevent garbage collection
    graph_label.pack()

# Function to show the loss graph
def show_graph_loss():
    global graph_loss_window
    if graph_loss_window is None:
        graph_loss_window = tk.Toplevel()
        graph_loss_window.title("Loss Graph")
        graph_loss_window.geometry("400x300")
    else:
        graph_loss_window.deiconify()
    
    # Load the graph image
    graph_image_path = r"E:\Developement\DeepLearning project\videoClassificationModel\model_loss.png"  # Corrected path
    graph_image = Image.open(graph_image_path)
    graph_image = graph_image.resize((400, 300))

    # Convert the PIL image to a Tkinter-compatible format
    graph_image_tk = ImageTk.PhotoImage(graph_image)

    # Display the graph image in the window
    graph_label = tk.Label(graph_loss_window, image=graph_image_tk)
    graph_label.image = graph_image_tk  # Keep a reference to prevent garbage collection
    graph_label.pack()

# Create the Tkinter window
window = tk.Tk()
window.title("Video Processing UI")

# Function to create a section with a title and a specified width
def create_section(title):
    section_frame = tk.Frame(window, bd=2, relief=tk.GROOVE)
    section_frame.pack(side="top", padx=10, pady=10, fill="both", expand=True)

    section_label = tk.Label(section_frame, text=title, font=("Arial", 12, "bold"))
    section_label.pack(pady=(10, 5))

    return section_frame

# Add a background image
bg_image = tk.PhotoImage(file="E:/Developement/DeepLearning project/deeplearningbg.png")
bg_label = tk.Label(window, image=bg_image)
bg_label.place(relwidth=1, relheight=1)

# Create sections for image, video, and graphs
image_section = create_section("Image Processing")
video_section = create_section("Video Processing")
graph_section = create_section("Graphs")

# Add buttons for image processing
select_image_button = tk.Button(image_section, text="Select Image", command=select_image)
select_image_button.pack(pady=(0, 10))

process_image_button = tk.Button(image_section, text="Process and Display Image", command=process_single_frame)
process_image_button.pack(pady=(0, 10))

# Add buttons for video processing
select_video_button = tk.Button(video_section, text="Select Video", command=select_video)
select_video_button.pack(pady=(0, 10))

process_video_button = tk.Button(video_section, text="Process and Display Video", command=process_and_display_video)
process_video_button.pack(pady=(0, 10))

# Add buttons for showing graphs
show_Graph_accuracy_button = tk.Button(graph_section, text="Show Accuracy Graph", command=show_graph_accuracy, bg="lightgreen")
show_Graph_accuracy_button.pack(pady=(0, 10))

show_Graph_loss_button = tk.Button(graph_section, text="Show Loss Graph", command=show_graph_loss, bg="red")
show_Graph_loss_button.pack(pady=(0, 10))

# Load the model and necessary objects when the window opens
load_model_and_objects()

# Run the Tkinter event loop
window.mainloop()
