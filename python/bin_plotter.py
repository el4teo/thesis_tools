import json
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.widgets import TextBox
import numpy as np
import tkinter as tk
from tkinter import filedialog

import logging

class bin_plotter:
    """Reads samples binary files and exports png images"""

    def __init__(self, filename = None):
        # Representation constant values
        self.MARGIN_LEFT = 0.04
        self.MARGIN_RIGHT = 0.03
        self.MARGIN_UP = 0.05
        self.MARGIN_DOWN = 0.05
        self.total_width = 1 - self.MARGIN_LEFT - self.MARGIN_RIGHT
        self.total_high = 1 - self.MARGIN_UP - self.MARGIN_DOWN

        # Default configuration values
        self.config_data = {}
        self.config_data["show_time"] = True
        self.config_data["show_fft"] = True
        self.config_data["hide_toolbar"] = False
        self.config_data["sampling_rate_MSps"] = 100
        self.config_data["amplitude_range"] = 1
        self.config_data["n_samples_to_read"] = 2000000 # -1 means all of them
        self.config_data["fft_xlin_min_Hz"] = 1
        self.config_data["fft_xlin_max_Hz"] = 30000000
        self.config_data["log_level"] = "INFO"
        self.config_data["show_figure"] = True
        self.config_data["export_png"] = True
        
        # Handlers
        self.ax_time = None
        self.ax_fft = None
        
        # File parameters
        self.config_file = "./python/bin_plotter_config.json"
        self.log_file = "./python/bin_plotter.log"
        self.filename = filename
        self.logger = logging.getLogger(__name__ + self.filename)
        
        # Data variables
        self.samples = None
        
        # Representation tasks
        self.read_conf_file()
        self.config_logging()
        self.config_fig()
        self.read_bin_file()
        self.plot_samples()

        if self.config_data["show_figure"]:
            plt.show()

        if self.config_data["export_png"]:
            self.export_png()
        
    def __del__(self):
        self.save_current_config()
        plt.close(self.fig)

    def config_logging(self):
        if self.config_data["log_level"] == "CRITICAL":
            log_level = logging.CRITICAL
        elif self.config_data["log_level"] == "ERROR":
            log_level = logging.ERROR
        elif self.config_data["log_level"] == "WARNING":
            log_level = logging.WARNING
        elif self.config_data["log_level"] == "INFO":
            log_level = logging.INFO
        elif self.config_data["log_level"] == "DEBUG":
            log_level = logging.DEBUG
        elif self.config_data["log_level"] == "NOTSET":
            log_level = logging.NOTSET
        else:
            raise ValueError(f"Invalid log level: {self.config_data["log_level"]}")
        
        self.logger.setLevel(log_level)
    
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        file_handler = logging.FileHandler(self.log_file, mode="a")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def export_png(self, filename = None):
        if filename is None:
            if not self.filename:
                raise ValueError("No file selected")
            else:
                filename = self.filename[:-4] + ".png"
            
        self.fig.savefig(filename, dpi=300)
        self.logger.info(f"Exported: {filename}")
        
    def plot_samples(self):
        if not self.config_data["show_time"] and not self.config_data["show_fft"]:
            return
        
        n_samples = len(self.samples)
        delta_x = 1 / (self.config_data["sampling_rate_MSps"] * 1000000)
        x_time = np.linspace(0, delta_x * n_samples - delta_x, n_samples)
        y_time = self.samples
        
        if self.config_data["show_time"]:
            self.ax_time.plot(x_time * 1000, y_time, color='#00009A')
            self.ax_time.set_xlabel("Time (ms)")
            if self.config_data["amplitude_range"] == 1:
                self.ax_time.set_ylabel("Amplitude (norm.)")
            else:
                self.ax_time.set_ylabel("Amplitude (V)")
            self.ax_time.set_xlim(0, (x_time[-1] + delta_x) * 1000)
            # self.ax_time.set_ylim(min(y_time) * 1.1, max(y_time) * 1.1)
        
        if self.config_data["show_fft"]:
            y_fft = np.fft.fft(y_time)
            y_fft = np.abs(y_fft[:n_samples // 2])  # Magnitud de la FFT (solo parte positiva)
            x_fft = np.fft.fftfreq(n_samples, d=delta_x)[:n_samples // 2]  # Frecuencias

            # Convertir a escala logarítmica (dB)
            y_fft_db = 20 * np.log10(y_fft + 1e-12)  # Se suma un pequeño valor para evitar log(0)

            # Configurar escala logarítmica en X y Y
            self.ax_fft.plot(x_fft, y_fft_db, color='#00009A')  
            self.ax_fft.set_xlabel("Frequency (Hz)")
            self.ax_fft.set_ylabel("Magnitude (dB)")
            self.ax_fft.set_xscale("log")
            # self.ax_fft.set_xlim(min(x_fft[x_fft > 0]), max(x_fft))  # Evita 0 en escala log
            fft_xlin_min_Hz = min(x_fft[x_fft > 0])
            fft_xlin_max_Hz = max(x_fft)
            if 0 < self.config_data["fft_xlin_min_Hz"] and self.config_data["fft_xlin_min_Hz"] < self.config_data["fft_xlin_max_Hz"]:
                fft_xlin_min_Hz = self.config_data["fft_xlin_min_Hz"]
                fft_xlin_max_Hz = self.config_data["fft_xlin_max_Hz"]

            self.ax_fft.set_xlim(fft_xlin_min_Hz, fft_xlin_max_Hz)  
            self.ax_fft.set_ylim(min(y_fft_db), max(y_fft_db) + 10)  
             
    def read_bin_file(self):
        if self.filename is None:
            root = tk.Tk()
            root.withdraw()
            self.filename = filedialog.askopenfilename(filetypes=[("Binary files", "*.bin")], title="Select the binary file")
        
        if not self.filename:
            raise ValueError("No file selected.")
        
        try:
            with open(self.filename, 'rb') as file:
                file.seek(0, 2)  # Move to end of file
                num_of_bytes = file.tell()
                file.seek(0, 0)  # Move to start of file
                
                # Define number of samples to read
                num_samples = num_of_bytes // 2  # int16 -> 2 bytes per sample
                if self.config_data["n_samples_to_read"] != -1:
                    num_samples = min(num_samples, self.config_data["n_samples_to_read"])
                
                # Read samples
                self.samples = np.fromfile(file, dtype=np.int16, count=num_samples)
                
                # Normalize samples from [-32768, 32767] to [-1, 1]
                self.samples = (self.samples + 32768) / (65535 / 2) - 1
                
                # Apply range factor
                if self.config_data["amplitude_range"] != 1:
                    self.samples *= self.config_data["amplitude_range"]
                
                self.logger.debug(f"Readen {num_samples} samples from {self.filename}")
        
        except Exception as e:
            raise RuntimeError(f"Error reading the file: {e}")
    
    def config_fig(self):
        if (self.config_data["hide_toolbar"]):
            self.fig.canvas.toolbar.pack_forget()
        if self.filename is None:
            self.fig = plt.figure(figsize=(8,4.5),num='Bin plotter')
        else:
            self.fig = plt.figure(figsize=(8,4.5),num=self.filename)
        self.mng = plt.get_current_fig_manager()
        
        self.config_axes()
        # self.plot_default_data()
        # self.config_buttons()
        # self.config_text_box()

    def read_conf_file(self):
        try:
            with open(self.config_file, "r") as config_file:
                self.config_data = json.load(config_file)
        except FileNotFoundError:
            pass
            # raise RuntimeError("The configuration file does not exist.")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Error loading configuration data: {e}")
        
    def config_axes(self):
        if self.config_data["show_time"]:
            if not self.config_data["show_fft"]:
                base_offset_x = self.MARGIN_LEFT + 0.05
                base_offset_y = self.MARGIN_DOWN + 0.08
                ax_wdith = self.total_width * 0.95
                ax_high = self.total_high * 0.87
            else:
                base_offset_x = self.MARGIN_LEFT + 0.05
                base_offset_y = self.MARGIN_DOWN + 0.08
                ax_wdith = self.total_width * 0.40
                ax_high = self.total_high * 0.87
            self.ax_time = self.fig.add_axes([base_offset_x, base_offset_y, ax_wdith, ax_high])
            # self.ax_time.set_xticks([0, 90, 180, 270, 360])
            self.ax_time.set_facecolor('lightgoldenrodyellow')
            # self.ax_time.set_xticklabels(['0', '', '', '', '360']) 
            self.ax_time.tick_params(axis='both',labelsize=8)
            self.ax_time.grid(visible=True, which='both', axis='both', linestyle='--')
            self.ax_time.set_title("TIME")
        
        if self.config_data["show_fft"]:
            if not self.config_data["show_time"]:
                base_offset_x = self.MARGIN_LEFT + 0.05
                base_offset_y = self.MARGIN_DOWN + 0.08
                ax_wdith = self.total_width * 1.0
                ax_high = self.total_high * 0.87
            else:
                base_offset_x = self.MARGIN_LEFT + 0.55
                base_offset_y = self.MARGIN_DOWN + 0.08
                ax_wdith = self.total_width * 0.40
                ax_high = self.total_high * 0.87
            self.ax_fft = self.fig.add_axes([base_offset_x, base_offset_y, ax_wdith, ax_high])
            self.ax_fft.set_ylim(-1.1, 1.1) 
            self.ax_fft.set_xlim(0, 360)
            self.ax_fft.set_xticks([0, 90, 180, 270, 360])
            self.ax_fft.set_facecolor('lightgoldenrodyellow')
            # self.ax_fft.set_xticklabels(['0', '', '', '', '360']) 
            self.ax_fft.tick_params(axis='both',labelsize=8)
            self.ax_fft.grid(visible=True, which='both', axis='both', linestyle='--')
            self.ax_fft.set_title("FFT")
            
    def plot_default_data(self):
        if self.config_data["show_time"]:
            n_points_sin = 100
            x_rad = np.linspace(0, 2 * np.pi, n_points_sin)
            x_deg = np.linspace(0, 360, n_points_sin)
            y_ref = np.sin(x_rad)
            sin_amp = 1
            sin_offset = 0
            extra_idx = 0
            margin_factor = 1.1
            y_sin = y_ref * sin_amp + sin_offset
            self.ax_time = self.ax_time.plot(x_deg, y_sin, color='#9A0000')
        
    def save_current_config(self):
        with open(self.config_file, "w", encoding="utf-8") as config_file:
            json.dump(self.config_data, config_file, indent=4)
        self.logger.debug(f"Configuration data has been written to {self.config_file}")

def find_bin_files(directory):
    """Recursively finds all .bin files in the given directory."""
    bin_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".bin"):
                bin_files.append(os.path.join(root, file))
    return bin_files

def png_already_exists(filename):
    return os.path.exists(filename[:-4] + ".png")

def main():
    directory = "./DB/Noise"
    if os.path.isdir(directory):
        bin_files = find_bin_files(directory)
        for file in bin_files:
            if not png_already_exists(file):
                bin_plotter(file)
    else:
        # Invalid directory
        bin_plotter()
    
if __name__ == "__main__":
    main()
