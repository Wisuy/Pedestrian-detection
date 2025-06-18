import cv2
import tkinter as tk
import time
import datetime

from ONNXDetector import ONNXDetector
from CameraFeed import CameraFeed
from FrameProcessor import FrameProcessor

from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
from pygrabber.dshow_graph import FilterGraph

def list_available_cameras():
    graph = FilterGraph()
    devices = graph.get_input_devices()
    return [(i, device) for i, device in enumerate(devices)]

class ApplicationGUI:
    def __init__(self, master):
        self.master = master
        master.title("Camera Feed")
        master.geometry("1150x600")

        self.cap = None
        self.processor = None
        self.running = False
        self.onnx_model = None
        self.car_deceleration_mps2 = [0.0]

        self._create_widgets()
        self.refresh_cameras()

    def _create_widgets(self):
        top_frame = tk.Frame(self.master)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=0, pady=0)

        settings_frame = tk.Frame(top_frame)
        settings_frame.pack(side=tk.LEFT, anchor='nw', padx=5, pady=5)

        camera_group = ttk.LabelFrame(settings_frame, text="Camera Selection", padding=(10, 5))
        camera_group.grid(row=0, column=0, columnspan=3, padx=5, pady=5, sticky="ew")
        camera_group.grid_columnconfigure(1, weight=1)

        tk.Label(camera_group, text="Select Camera(VGA):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.cameras_info = []
        self.camera_names = []
        
        self.selected_camera_name = tk.StringVar()
        self.combo = ttk.Combobox(camera_group, values=self.camera_names, textvariable=self.selected_camera_name, state="readonly", width=30)
        self.combo.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        tk.Label(camera_group, text="IP Camera URL(http):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.ip_camera_url_var = tk.StringVar(value="http://192.168./video")
        self.ip_camera_url_entry = tk.Entry(camera_group, textvariable=self.ip_camera_url_var, width=30)
        self.ip_camera_url_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        if self.selected_camera_name.get() != "IP Camera":
            self.ip_camera_url_entry.config(state=tk.DISABLED)

        self.selected_camera_name.trace_add("write", self._on_camera_selection_change)

        refresh_button = ttk.Button(camera_group, text="Refresh Cameras", command=self.refresh_cameras)
        refresh_button.grid(row=0, column=2, padx=5, pady=5, sticky="e")

        # Model Parameters Group
        model_params_group = ttk.LabelFrame(settings_frame, text="Parameters", padding=(10, 5))
        model_params_group.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="ew")
        model_params_group.grid_columnconfigure(1, weight=1)

        tk.Label(model_params_group, text="Confidence:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.conf_var = tk.DoubleVar(value=0.5)
        conf_slider = tk.Scale(model_params_group, from_=0.1, to=0.9, resolution=0.01, orient=tk.HORIZONTAL, variable=self.conf_var, length=180)
        conf_slider.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        tk.Label(model_params_group, text="Average Height (meters):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.avg_height_var = tk.DoubleVar(value=1.7)
        avg_height_slider = tk.Scale(model_params_group, from_=1.5, to=2, resolution=0.01, orient=tk.HORIZONTAL, variable=self.avg_height_var, length=180)
        avg_height_slider.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        tk.Label(model_params_group, text="Max Braking G-Force:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.max_g_force_var = tk.DoubleVar(value=0.787)
        max_g_force_slider = tk.Scale(model_params_group, from_=0.2, to=1.2, resolution=0.01, orient=tk.HORIZONTAL, variable=self.max_g_force_var, length=180)
        max_g_force_slider.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        tk.Label(model_params_group, text="Focal Length:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.focal_length_var = tk.DoubleVar(value=605)
        focal_length_entry = tk.Entry(model_params_group, textvariable=self.focal_length_var, width=15)
        focal_length_entry.grid(row=3, column=1, padx=5, pady=5, sticky="ew")

        tk.Label(model_params_group, text="Model Path:").grid(row=4, column=0, padx=5, pady=5, sticky="w")
        self.model_path_var = tk.StringVar(value="model.onnx")
        model_path_entry = tk.Entry(model_params_group, textvariable=self.model_path_var, width=30)
        model_path_entry.grid(row=4, column=1, padx=5, pady=5, sticky="ew")

        browse_button = ttk.Button(model_params_group, text="Browse", command=self._browse_model_path)
        browse_button.grid(row=4, column=2, padx=5, pady=5, sticky="e")

        self.show_labels_var = tk.BooleanVar(value=True)
        show_labels_checkbutton = tk.Checkbutton(model_params_group, text="Show Labels", variable=self.show_labels_var)
        show_labels_checkbutton.grid(row=5, column=0, columnspan=3, padx=5, pady=5, sticky="w")

        # Video Frame and Stats
        video_frame = tk.Frame(top_frame)
        video_frame.pack(side=tk.RIGHT, anchor='ne', padx=5, pady=5)

        self.video_label = tk.Label(video_frame)
        self.video_label.pack()

        stats_frame = tk.Frame(video_frame)
        stats_frame.pack(pady=5)

        self.pre_time_var = tk.StringVar(value="")
        self.inf_time_var = tk.StringVar(value="")
        self.post_time_var = tk.StringVar(value="")
        self.avg_min_speed_var = tk.StringVar(value="") 
        self.res_var = tk.StringVar(value="")

        tk.Label(stats_frame, textvariable=self.pre_time_var).grid(row=0, column=0, padx=10)
        tk.Label(stats_frame, textvariable=self.inf_time_var).grid(row=0, column=1, padx=10)
        tk.Label(stats_frame, textvariable=self.post_time_var).grid(row=0, column=2, padx=10)
        tk.Label(stats_frame, textvariable=self.avg_min_speed_var).grid(row=0, column=3, padx=10) 
        tk.Label(stats_frame, textvariable=self.res_var).grid(row=0, column=4, padx=10) 

        # Control Buttons
        bottom_frame = tk.Frame(self.master)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        self.start_btn = ttk.Button(bottom_frame, text="Start Feed", command=self.start_feed)
        self.start_btn.pack(side=tk.LEFT, padx=5, pady=10)
        self.stop_btn = ttk.Button(bottom_frame, text="Stop Feed", command=self.stop_feed, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5, pady=10)
        
        self.save_output_btn = ttk.Button(bottom_frame, text="Save Current Frame", command=self._save_output_frame)
        self.save_output_btn.pack(side=tk.LEFT, padx=5, pady=10)

        self.master.protocol("WM_DELETE_WINDOW", self.on_close)

    def _on_camera_selection_change(self, *args):
        if self.selected_camera_name.get() == "IP Camera":
            self.ip_camera_url_entry.config(state=tk.NORMAL)
            self.ip_camera_url_var.set("http://192.168./video")
        else:
            self.ip_camera_url_entry.config(state=tk.DISABLED)
            self.ip_camera_url_var.set("")

    def refresh_cameras(self):
        self.cameras_info = list_available_cameras()
        self.camera_names = [info[1] for info in self.cameras_info]
        self.camera_names.insert(0, "IP Camera")
        self.combo['values'] = self.camera_names
        if self.selected_camera_name.get() not in self.camera_names:
            self.selected_camera_name.set(self.camera_names[0] if self.camera_names else "")

    def _browse_model_path(self):
        filepath = filedialog.askopenfilename(
            title="Select ONNX Model",
            filetypes=[("ONNX files", "*.onnx"), ("All files", "*.*")]
        )
        if filepath:
            self.model_path_var.set(filepath)

    def start_feed(self):
        if self.cap is not None:
            self.cap.stop()
            self.cap = None 
        if self.processor is not None:
            self.processor.stop()
            self.processor = None

        model_path = self.model_path_var.get()
        if not model_path:
            messagebox.showerror("Error", "Please select a model path.")
            return

        conf_threshold = self.conf_var.get()
        avg_height = self.avg_height_var.get()
        focal_length = self.focal_length_var.get()
        
        max_g_force = self.max_g_force_var.get()
        self.car_deceleration_mps2[0] = max_g_force * 9.80665 # Convert G to m/s^2

        try:
            self.onnx_model = ONNXDetector(model_path, conf_threshold=conf_threshold,
                                        real_height=avg_height, focal_length=focal_length)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
            return

        selected_source = self.selected_camera_name.get()
        if selected_source == "IP Camera":
            ip_url = self.ip_camera_url_var.get().strip()
            if not ip_url:
                messagebox.showerror("Error", "Please enter an IP Camera URL.")
                return
            self.cap = CameraFeed(ip_url)
        else:
            camera_index = -1
            for i, name in self.cameras_info:
                if name == selected_source:
                    camera_index = i
                    break

            if camera_index == -1:
                messagebox.showerror("Error", f"Could not find index for camera: {selected_source}")
                return
            self.cap = CameraFeed(camera_index)
        
        self.cap.start()

        time.sleep(2) 

        initial_ret, initial_frame = self.cap.read_latest_raw_frame()
        if not initial_ret or initial_frame is None:
            messagebox.showerror("Error", f"Could not open camera from source: {selected_source}. Please ensure the camera is available and not in use by another application.")
            self.cap.stop()
            return

        self.processor = FrameProcessor(self.cap, self.onnx_model, self.show_labels_var, self.car_deceleration_mps2)
        self.processor.start()
        
        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.update_frame()

    def stop_feed(self):
        self.running = False

        if self.processor is not None:
            self.processor.stop() 
            self.processor = None
        if self.cap is not None:
            self.cap.stop() 
            self.cap = None
        
        self.video_label.config(image='', text='')
        self.pre_time_var.set("")
        self.inf_time_var.set("")
        self.post_time_var.set("")
        self.avg_min_speed_var.set("") 
        self.res_var.set("")
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def _save_output_frame(self):
        if self.processor is None or not self.running:
            messagebox.showwarning("Warning", "No active camera feed to save output from.")
            return

        frame, info = self.processor.get_latest_processed_frame()
        
        if frame is None:
            messagebox.showwarning("Warning", "No processed frame available to save.")
            return

        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detection_output_{timestamp}.png"
            
            cv2.imwrite(filename, frame)
            messagebox.showinfo("Success", f"Output saved successfully as {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save output: {e}")

    def update_frame(self):
        if not self.running or self.processor is None:
            self.video_label.config(text="Camera feed stopped or not initialized.")
            return
        
        frame, info = self.processor.get_latest_processed_frame()
        
        if frame is None:
            self.video_label.config(text="Waiting for processed frame...")
            self.master.after(100, self.update_frame)
            return

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            self.master.title(f"Camera Feed")

            self.pre_time_var.set(f"Pre: {info.get('preprocess_time', 0):.1f} ms")
            self.inf_time_var.set(f"Inference: {info.get('inference_time', 0):.1f} ms")
            self.post_time_var.set(f"Post: {info.get('postprocess_time', 0):.1f} ms")
            self.avg_min_speed_var.set(f"Min Speed: {info.get('avg_min_speed_kph', 0):.2f} km/h") 
            self.res_var.set(f"Resolution: {info.get('resolution', 'N/A')}")

        except Exception as e:
            error_message = f"Error during frame display: {e}"
            print(error_message)
            self.video_label.config(text=error_message)

        self.master.after(10, self.update_frame)

    def on_close(self):
        self.stop_feed()
        self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ApplicationGUI(root)
    root.mainloop()