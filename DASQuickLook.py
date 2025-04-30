import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.integrate import cumulative_trapezoid
from scipy.io import loadmat
import plotly.graph_objects as go 
import xml.etree.ElementTree as ET
import os
import struct
import re
import h5py

class DASQuickLookApp:
    def __init__(self, master):
        self.master = master
        master.title("DASQuickLook")

        # Initialize variables
        self.pname = ''
        self.tests = []
        self.filtfreq = 1000
        self.initial_condition_integral = 0

        # Create GUI elements
        self.create_menu()
        self.create_widgets()  

        self.flush_vars()      

    def flush_vars(self):
        """Resets all variables to their default states."""
        # Initialize variables
        self.data = np.array([])
        self.hdr = []
        self.testname = ''
        self.file_type = None
        self.current_selections = []
        self.plot_channels = []
        self.max = []
        self.min = []
        self.lgnd = None
        self.filtfreq = 1000
        self.initial_condition_integral = 0
        self.b = []
        self.a = []
        self.freq = []
        self.mag = []
        self.t_integ = []
        self.frst_integ = []
        self.scnd_integ = []
        self.tat_results = {} 
        self.points = []
        self.slope_results = {}
        self.cursor = None
        self.active_zoom_axis = None
        self.filter_active = False
        self.multiplier_active = False
        if hasattr(self, 'original_data'):
            del self.original_data
        
        # Reset zoom state
        self.zoom_x_activated = False 
        self.zoom_y_activated = False 
        self.fig2_zoom_x_activated = False 
        self.fig2_zoom_y_activated = False
        self.zoom_rect = None

        # Reset button states (and text if necessary)
        self.plot_button.config(state='disabled')
        self.reset_filter_button.config(state='disabled')
        self.normalize_toggle.config(state='disabled')
        self.filter_button.config(state='disabled')
        self.filter_magnitude_button.config(state='disabled')
        self.fft_button.config(state='disabled', text='FFT')
        self.plotly_button.config(state='disabled')
        self.zoom_x_button.config(state='disabled', text='Zoom X')
        self.zoom_y_button.config(state='disabled', text='Zoom Y')
        self.first_integral_button.config(state='disabled', text='1st Integral')
        self.second_integral_button.config(state='disabled', text='2nd Integral')
        self.save_button.config(state='disabled')
        self.xhairs_button.config(state='disabled', text='X-hairs')
        self.ax2_xhairs_button.config(state='disabled', text='X-hairs')
        self.slope_button.config(state='disabled', text='2-pt Slope')
        self.tat_button.config(state='disabled')
        self.xplot_toggle.config(state='disabled')
        self.dualy_toggle.config(state='disabled')
        self.fig2_zoom_x_button.config(state='disabled', text='Zoom X')
        self.fig2_zoom_y_button.config(state='disabled', text='Zoom Y')

    def create_menu(self):
        # Create menubar
        menubar = tk.Menu(self.master)
        self.master.config(menu=menubar)

        # File menu
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open Directory", command=self.open_directory)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.master.quit)
        menubar.add_cascade(label="File", menu=filemenu)

    def create_widgets(self):
        # Configure main window resizing behavior
        self.master.rowconfigure(0, weight=1)
        self.master.columnconfigure(0, weight=1)

        # Main frame
        mainframe = tk.Frame(self.master)
        mainframe.grid(row=0, column=0, sticky="nsew")
        # mainframe.pack(fill='both', expand=True)

        # Configure mainframe grid weights
        mainframe.rowconfigure(0, weight=0)  # Top row (test and channel selection) doesn't expand
        mainframe.rowconfigure(1, weight=1)  # Bottom row (plot and controls) expands
        mainframe.columnconfigure(0, weight=1)  # Plot area expands
        mainframe.columnconfigure(1, weight=0)  # Controls don't expand

        # Create top frame for test and channel selection
        top_frame = tk.Frame(mainframe)
        top_frame.grid(row=0, column=0, columnspan=2, sticky='ew', padx=5, pady=5)
    
        # Left frame for test and channel selection
        left_frame = tk.Frame(top_frame)
        left_frame.pack(side='left', fill='x')

        # Bottom frame for plot and controls
        bottom_frame = tk.Frame(mainframe)
        bottom_frame.grid(row=1, column=0, columnspan=2, sticky='nsew', padx=5, pady=1)

        # Controls frame
        controls_frame = tk.Frame(bottom_frame)
        controls_frame.pack(side='right', fill='y')

        # Test Selection
        test_frame = tk.LabelFrame(top_frame, text="Test Selection")
        test_frame.pack(side='left', fill='x', expand=False, padx=5)

            # Frame for the Open buttons
        open_buttons_frame = tk.Frame(test_frame)
        open_buttons_frame.pack(pady=2)  # Add some padding around the buttons

            # Configure column weights to make buttons expand horizontally
        open_buttons_frame.columnconfigure(0, weight=1)
        open_buttons_frame.columnconfigure(1, weight=1)

            # Open buttons
        self.open_button = ttk.Button(open_buttons_frame, text="Open Directory (raw)", command=self.open_directory)
        self.open_button.grid(row=0, column=0, sticky="ew", padx=2)

        self.open_mat_button = ttk.Button(open_buttons_frame, text="Open Directory (.mat)", command=self.open_mat_directory)
        self.open_mat_button.grid(row=0, column=1, sticky="ew", padx=2)

            # Combobox below the buttons
        self.test_combobox = ttk.Combobox(test_frame, state='readonly', width=50)
        self.test_combobox.pack(pady=2)
        self.test_combobox.bind('<<ComboboxSelected>>', self.select_test)

        # Filter controls
        filter_frame = tk.LabelFrame(controls_frame, text="Filter (Hz)")
        filter_frame.pack(pady=2)

            # Create top frame for entry and apply button
        top_filter_frame = tk.Frame(filter_frame)
        top_filter_frame.pack(fill='x', pady=2)

        self.filter_entry = ttk.Entry(top_filter_frame, width=15)
        self.filter_entry.insert(0, str(self.filtfreq))
        self.filter_entry.pack(side='left', fill='x', expand=True, padx=3)

            # Create frame for buttons to control their width
        button_frame = tk.Frame(filter_frame)
        button_frame.pack(fill='x', padx=3)

            # Add both buttons with same width
        button_width = 20  # Adjust this value as needed

        self.filter_button = ttk.Button(button_frame, text="Apply Filter", command=self.apply_filter, state='disabled', width=button_width)
        self.filter_button.pack(fill='x', pady=2)

            # Filter & Magnitude Button
        self.filter_magnitude_button = ttk.Button(filter_frame, text="Filter & Multiplier", command=self.open_filter_magnitude_window, state='disabled', width=button_width)
        self.filter_magnitude_button.pack(fill='x', pady=2)  

        self.reset_filter_button = ttk.Button(button_frame, text="Reset Filter", command=self.reset_filter, state='disabled', width=button_width)
        self.reset_filter_button.pack(fill='x', pady=(0, 2))

        # Channel selection
        channel_frame = tk.LabelFrame(top_frame, text="Channel Selection")
        channel_frame.pack(side='left', fill='both', expand=True, padx=5)

            # Create scrollbar first
        scrollbar = ttk.Scrollbar(channel_frame, orient='vertical')
        scrollbar.pack(side='right', fill='y')

            # Create listbox and connect to scrollbar
        self.channel_listbox = tk.Listbox(channel_frame, selectmode='multiple', height=6, width=30, yscrollcommand=scrollbar.set)
        self.channel_listbox.pack(side='left', fill='both', expand=True, pady=2)

            # Configure scrollbar to scroll listbox
        scrollbar.config(command=self.channel_listbox.yview)

            # Bind selection event
        self.channel_listbox.bind('<<ListboxSelect>>', self.select_channels)

        # Plot button
        plot_info_frame = tk.Frame(top_frame)
        plot_info_frame.pack(side='left', fill='x', expand=True, padx=5)

        self.plot_button = ttk.Button(plot_info_frame, text="Plot", command=self.plot_data, state='disabled')
        self.plot_button.pack(side='left')

        # Info listbox
            # Info listbox Frame
        info_frame = tk.LabelFrame(plot_info_frame, text="Peaks (Plotted Data)")  # Use LabelFrame for border
        info_frame.pack(side='left', fill='both', expand=True, padx=5)

            # Create scrollbar for info listbox
        info_scrollbar_y = ttk.Scrollbar(info_frame, orient='vertical')
        info_scrollbar_y.pack(side='right', fill='y')
        info_scrollbar_x = ttk.Scrollbar(info_frame, orient='horizontal')
        info_scrollbar_x.pack(side='bottom', fill='x')

            # Headers for info listbox (using a separate frame for styling)
        header_frame = tk.Frame(info_frame)
        header_frame.pack(side='top', fill='x')  # Pack at the top

        channel_header = tk.Label(header_frame, text="Channel", font=('TkDefaultFont', 8, 'bold'), width=10)
        channel_header.pack(side='left')
        max_header = tk.Label(header_frame, text="Max", font=('TkDefaultFont', 8, 'bold'), width=10)
        max_header.pack(side='left')
        min_header = tk.Label(header_frame, text="Min", font=('TkDefaultFont', 8, 'bold'), width=10)
        min_header.pack(side='left')
        threshold_header = tk.Label(header_frame, text="Threshold", font=('TkDefaultFont', 8, 'bold'), width=10)
        threshold_header.pack(side='left')
        tat_header = tk.Label(header_frame, text="TAT", font=('TkDefaultFont', 8, 'bold'), width=10)
        tat_header.pack(side='left')
        slope_header = tk.Label(header_frame, text="2-pt Slope", font=('TkDefaultFont', 8, 'bold'), width=10)
        slope_header.pack(side='left')
        dx_header = tk.Label(header_frame, text="dx\n(sec)", font=('TkDefaultFont', 8, 'bold'), width=10)
        dx_header.pack(side='left')
        dy_header = tk.Label(header_frame, text="dy\n(chn units)", font=('TkDefaultFont', 8, 'bold'), width=10)
        dy_header.pack(side='left')

        # Separator line below headers
        separator = ttk.Separator(info_frame, orient='horizontal')
        separator.pack(side='top', fill='x')

            # Info listbox itself
        self.info_listbox = tk.Listbox(info_frame, xscrollcommand=info_scrollbar_x.set, yscrollcommand=info_scrollbar_y.set, height=6, width=30)
        self.info_listbox.pack(side='left', fill='both', expand=True, pady=2)

            # Configure scrollbar to scroll info listbox
        info_scrollbar_y.config(command=self.info_listbox.yview)
        info_scrollbar_x.config(command=self.info_listbox.xview)

        # Matplotlib figure and canvas
        self.fig = plt.Figure(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=bottom_frame)
        self.canvas.get_tk_widget().pack(side='left', fill='both', expand=True)
        self.ax1 = self.fig.add_subplot(111)

        # Normalize toggle
        self.normalize_var = tk.BooleanVar(value=False)
        self.normalize_toggle = ttk.Checkbutton(controls_frame, text="Normalize", variable=self.normalize_var, command=self.plot_data, state='disabled')
        self.normalize_toggle.pack(pady=2)

        # Multiple Channel Buttons
        mult_chan_frame = tk.Frame(controls_frame)
        mult_chan_frame.pack(pady=2)

            # Cross Plot Toggle
        self.xplot_var = tk.BooleanVar(value=False)
        self.xplot_toggle = ttk.Checkbutton(mult_chan_frame, text="X-Plot", variable=self.xplot_var, command=self.plot_data, state='disabled')
        self.xplot_toggle.pack(side=tk.LEFT, padx=2)

            # Double Y Toggle
        self.dualy_var = tk.BooleanVar(value=False)
        self.dualy_toggle = ttk.Checkbutton(mult_chan_frame, text="Dual Y Plot", variable=self.dualy_var, command=self.plot_data, state='disabled')
        self.dualy_toggle.pack(side=tk.LEFT, padx=2)
                
        # Interaction Controls
        interaction_frame = tk.LabelFrame(controls_frame, text="Interactions")
        interaction_frame.pack(pady=0)

        # Figure 1 Interaction Frame
        fig1_interaction_frame = tk.LabelFrame(interaction_frame, text="Figure 1")
        fig1_interaction_frame.pack(side=tk.LEFT, pady=2, padx=2)  # Pack to the LEFT

            # X-hair Button (Figure 1)
        self.xhairs_button = ttk.Button(fig1_interaction_frame, text="X-hairs", command=self.toggle_xhairs, state='disabled')
        self.xhairs_button.pack(pady=2, padx=2)

            # Zoom X Button (Figure 1)
        self.zoom_x_button = ttk.Button(fig1_interaction_frame, text="Zoom X", command=self.activate_zoom_x, state=tk.DISABLED)
        self.zoom_x_button.pack(pady=2, padx=2)

            # Zoom Y Button (Figure 1)
        self.zoom_y_button = ttk.Button(fig1_interaction_frame, text="Zoom Y", command=self.activate_zoom_y, state=tk.DISABLED)
        self.zoom_y_button.pack(pady=2, padx=2)

        # Figure 2 Interaction Frame
        fig2_interaction_frame = tk.LabelFrame(interaction_frame, text="Figure 2")
        fig2_interaction_frame.pack(side=tk.LEFT, pady=2, padx=2)  # Pack to the LEFT

            # ax2 X-hairs Button (Figure 2)
        self.ax2_xhairs_button = ttk.Button(fig2_interaction_frame, text="X-hairs", command=self.toggle_ax2_xhairs, state='disabled')
        self.ax2_xhairs_button.pack(pady=2, padx=2)

            # Zoom X Button (Figure 2)
        self.fig2_zoom_x_button = ttk.Button(fig2_interaction_frame, text="Zoom X", command=self.activate_fig2_zoom_x, state=tk.DISABLED)
        self.fig2_zoom_x_button.pack(pady=2, padx=2)

            # Zoom Y Button (Figure 2)
        self.fig2_zoom_y_button = ttk.Button(fig2_interaction_frame, text="Zoom Y", command=self.activate_fig2_zoom_y, state=tk.DISABLED)
        self.fig2_zoom_y_button.pack(pady=2, padx=2) 

        # FFT Controls
        fft_frame = tk.LabelFrame(controls_frame, text="FFT")
        fft_frame.pack(pady=1)

        # FFT button
        self.fft_button = ttk.Button(fft_frame, text="FFT", command=self.toggle_fft, state='disabled')
        self.fft_button.pack(pady=0)

        # Integration Controls
        integration_box_frame = tk.LabelFrame(controls_frame, text="Integration")
        integration_box_frame.pack(pady=0)

        # Integration Buttons Frame 
        integration_frame = tk.Frame(integration_box_frame) # Create a frame for integration buttons
        integration_frame.pack(pady=2) # Pack below FFT zoom frame

        self.first_integral_button = ttk.Button(integration_frame, text="1st Integral", command=self.toggle_first_integral, state='disabled')
        self.first_integral_button.pack(side=tk.LEFT, padx=2) # Pack side-by-side

        self.second_integral_button = ttk.Button(integration_frame, text="2nd Integral", command=self.calculate_second_integral, state='disabled')
        self.second_integral_button.pack(side=tk.LEFT, padx=2) # Pack side-by-side

        self.decel_var = tk.BooleanVar(value=False)
        self.decel_checkbox = ttk.Checkbutton(integration_box_frame, text="Deceleration", variable=self.decel_var)
        self.decel_checkbox.pack(pady=2)

        # Initial condition controls
        initial_condition_frame = tk.Frame(integration_box_frame)
        initial_condition_frame.pack(pady=(0, 2))  # Pack below integration buttons

        self.initial_condition_entry = ttk.Entry(initial_condition_frame, width=10)
        self.initial_condition_entry.insert(0, str(self.initial_condition_integral))
        self.initial_condition_entry.pack(side=tk.LEFT, padx=2)

        self.integral_type = tk.StringVar(value="ft/s")  # Default to "normal"
        self.mps_radio = ttk.Radiobutton(initial_condition_frame, text="m/s", variable=self.integral_type, value="m/s")
        self.mps_radio.pack(side=tk.LEFT, padx=2)

        self.fps_radio = ttk.Radiobutton(initial_condition_frame, text="ft/s", variable=self.integral_type, value="ft/s")
        self.fps_radio.pack(side=tk.LEFT, padx=2)

        initial_condition_label = tk.Label(integration_box_frame, text="1st Integral Initial Condition")
        initial_condition_label.pack()

        # Slope and TAT button frame
        slope_tat_frame = tk.Frame(controls_frame)
        slope_tat_frame.pack(pady=5)

        # Slope button
        self.slope_button = ttk.Button(slope_tat_frame, text="2-pt Slope", command=self.calculate_slope, state='disabled')
        self.slope_button.pack(side=tk.LEFT, padx=2)

        # TAT button
        self.tat_button = ttk.Button(slope_tat_frame, text="TAT", command=self.calculate_tat, state='disabled')
        self.tat_button.pack(side=tk.LEFT, padx=2)

        # Plotly and Save button frame
        plotly_save_frame = tk.Frame(controls_frame)
        plotly_save_frame.pack(pady=5)

        # Plotly button
        self.plotly_button = ttk.Button(plotly_save_frame, text="Plotly", command=self.plot_to_plotly, state='disabled')
        self.plotly_button.pack(side=tk.LEFT, padx=2)

        # Save figure button
        self.save_button = ttk.Button(plotly_save_frame, text="Save Figure", command=self.save_current_figure, state='disabled')
        self.save_button.pack(side=tk.LEFT, padx=2)


        # Make sure the plot area expands when the window is resized
        mainframe.rowconfigure(0, weight=1)  # Row 0 (plot area) expands vertically
        mainframe.columnconfigure(1, weight=1) # Column 1 (plot area) expands horizontally 

    def open_directory(self):
        """Opens a directory and populates the test combobox."""
        # Open directory dialog
        tk.messagebox.showwarning("Warning", "Only .syn .dts .tlf is supported")
        self.pname = filedialog.askdirectory()

        if not self.pname:
            return  # No directory selected
        
        # Initialize dictionary to store test folders and their full paths
        self.tests = {}
        
        # Search through all subdirectories
        for root, dirs, files in os.walk(self.pname):
            for file in files:
                if file.lower().endswith(('.syn', '.dts', '.tlf')):
                    rel_path = os.path.relpath(root, self.pname)
                    if rel_path != '.':  # Don't add the root directory itself
                        # Create display name (folder name)
                        display_name = os.path.basename(rel_path)

                    file_path = os.path.join(root, file)
                    self.tests[display_name] = {'path': file_path, 'full_info': file_path, 'filename': file}

        # Update combobox
        if self.tests:
            # Set combobox values with full info and sort display names
            self.test_combobox['values'] = sorted(self.tests.keys())

            # Clear current selection
            self.test_combobox.set('')

    def open_mat_directory(self):
        tk.messagebox.showwarning("Warning", "Only .mat files v5 & v7.3 with 'data' and 'hdr' is supported")
        self.pname = filedialog.askdirectory()

        if not self.pname:
            return  # No directory selected
    
        self.tests = {}

        for file in os.listdir(self.pname):
            if file.lower().endswith('.mat'):
                file_path = os.path.join(self.pname, file)
                self.tests[file[:-4]] = {'path': file_path, 'full_info': file}  # Using file name without .mat as key

            # Update combobox
            if self.tests:
                self.test_combobox['values'] = sorted(self.tests.keys())
                self.test_combobox.set('')

    def select_test(self, event):
        """Handles test selection from the combobox."""
        if hasattr(self, 'ax2'):
            tk.messagebox.showwarning("2nd Figure Active", "Please remove 2nd figure and try again.")
            return
        
        self.deactivate_interactions()
        self.flush_vars()

        # Get selected test info
        selected_test = self.test_combobox.get()
        if not selected_test:
            return  # No test selected
        
        self.testname = selected_test
        file_path = os.path.join(self.pname, self.tests[self.testname]['path'])
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext == '.mat':
            self.file_type = 'MAT_FILE'
            self.load_mat_data()
        elif file_ext in ('.tlf', '.dts', '.syn'):
            if file_ext == '.tlf':
                self.file_type = 'TLF_FILE'
            elif file_ext == '.dts':
                self.file_type = 'DTS_FILE'
            else:
                self.file_type = 'SYN_FILE'
            self.load_data()
        else:
            tk.messagebox.showwarning("Unsupported File", f"File type {file_ext} is not supported.")
            return

        # Update channel listbox
        self.update_channel_listbox()

        self.normalize_var.set(False)
        self.xplot_var.set(False)
        self.dualy_var.set(False)

    def restore_channel_selections(self):
        if not self.channel_listbox.curselection():
            if hasattr(self, 'current_selections'):
                for idx in self.current_selections:
                    self.channel_listbox.selection_set(idx) 
                    self.channel_listbox.see(idx)  # Ensure visibility

    def load_mat_data(self):
        filename = self.tests[self.testname]['path']
        self.data = np.array([])  # Initialize data to empty array
        self.hdr = []      # Initialize hdr to empty list
        HDR = [[]]

        try:
            # First, try loading with h5py (for v7.3 files)
            with h5py.File(filename, 'r') as f:
                mat_data = np.array(f['data']).T.astype(float)

                hdr_refs = f['hdr']  # Access the dataset containing the references
                hdr_refs_np = np.array(hdr_refs) # Convert the HDF5 dataset to a NumPy array
                hdr_strings = np.empty(hdr_refs_np.shape, dtype=object) # Pre-allocate array of strings

                if hdr_refs_np.ndim > 2: # Check for > 2D
                    tk.messagebox.showwarning("HDR Dimensions", "HDR must be 1D or 2D")
                    return
                
                for i in range(hdr_refs_np.shape[0]):
                    for j in range(hdr_refs_np.shape[1]):
                        ref = hdr_refs_np[i, j]
                        hdr_dataset = f[ref]
                        hdr_data = np.array(hdr_dataset)
                        decoded_string = "".join([chr(c) for c in hdr_data.flatten()])
                        hdr_strings[i, j] = decoded_string
                
                if hdr_refs_np.shape[0] == 2:
                    hdr_units = hdr_strings[1,:].reshape(hdr_refs_np.shape[1])
                
                hdr_strings = hdr_strings[0,:].reshape(hdr_refs_np.shape[1])        
                mat_hdr = hdr_strings

            for k in range(mat_hdr.shape[0]):
                    if k == 0: 
                        HDR = [[mat_hdr[k]]]
                    else:
                        HDR.append([mat_hdr[k]])    
            self.hdr = HDR
            self.data = mat_data

        except OSError as e:  # Catch OSError (for incorrect file format)
            try:
                # If h5py fails, try loadmat (for older versions)
                mat = loadmat(filename, squeeze_me=True)
                mat_data = mat['data'].astype(float)
                if mat['hdr'].ndim == 2:
                    mat_hdr = mat['hdr'][:,0]
                    hdr_units = mat['hdr'][:,1]
                elif mat['hdr'].ndim == 1:
                    mat_hdr = mat['hdr']
                for k in range(mat_hdr.shape[0]):
                    if k == 0: 
                        HDR = [[mat_hdr[k]]]
                    else:
                        HDR.append([mat_hdr[k]])

                self.hdr = HDR
                self.data = mat_data
            except Exception as e:
                tk.messagebox.showwarning("Load MAT Failed", f"{e}")
                return
                
        except FileNotFoundError:
            tk.messagebox.showwarning("Error:", f"File not found at: {filename}")
            return

    def load_data(self):
        """Loads data based on the detected file type."""
        # Get the full directory path
        test_dir = os.path.join(self.pname, self.testname)

        if self.file_type == 'TLF_FILE':
            file = [f for f in os.listdir(test_dir) if f.lower().endswith('.tlf')]
            self.data, self.hdr = self.import_tlf(os.path.join(test_dir, file[0]))
        elif self.file_type == 'DTS_FILE':
            file = [f for f in os.listdir(test_dir) if f.lower().endswith('.dts')]
            self.data, self.hdr = self.import_dts(os.path.join(test_dir, file[0]))
        elif self.file_type == 'SYN_FILE':
            file = [f for f in os.listdir(test_dir) if f.lower().endswith('.set')]
            self.data, self.hdr = self.import_syn(os.path.join(test_dir, file[0]))
        else:
            tk.messagebox.showwarning("Unsupported File", "Data loading not implemented for this file type.")
            return

        # Apply DC offset correction (if data is loaded)
        if len(self.data) > 0:
            tmp = np.where(self.data[:, 0] <= 0.0)[0]
            for jj in range(1, self.data.shape[1]):
                avg = np.mean(self.data[tmp, jj])
                self.data[:, jj] -= avg

    def update_channel_listbox(self):
        # Update channel listbox with loaded header information
        self.channel_listbox.delete(0, 'end')
        for i in range(1, len(self.hdr)):
            self.channel_listbox.insert('end', self.hdr[i][0])
        
    def select_channels(self, event):
        self.deactivate_interactions()
        # Store the new selections
        if self.channel_listbox.curselection():
            self.current_selections = self.channel_listbox.curselection()
            self.plot_button.config(state='normal')
        else:
            self.plot_button.config(state='disabled')
        # Update plot_channels
        self.plot_channels = [i + 1 for i in self.current_selections]
        
        # Enable plot button if channels are selected
        if len(self.plot_channels) == 2:
            self.xplot_toggle.config(state='normal')
            self.dualy_toggle.config(state='normal')
        else:
            self.xplot_toggle.config(state='disabled')
            self.dualy_toggle.config(state='disabled')   

    def plot_data(self):
        self.restore_channel_selections()
        self.deactivate_interactions()

        # Check if any channels are selected
        if not self.channel_listbox.curselection():
            tk.messagebox.showwarning("No Channels Selected", "Please select channels")
            return  # Exit the function if no channels are selected
        
        if self.xhairs_button.cget('text') == 'Exit X-hairs':
            self.canvas.mpl_disconnect(self.motion_cid)
            self.xhairs_button.config(text='X-hairs')
            self.cursor.disconnect()
            del self.cursor
            self.canvas.draw()

        if self.ax2_xhairs_button.cget('text') == 'Exit X-hairs':
            self.canvas.mpl_disconnect(self.motion_cid)
            self.ax2_xhairs_button.config(text='X-hairs')
            self.cursor.disconnect()
            del self.cursor
            self.canvas.draw()
        
        if len(self.channel_listbox.curselection()) != 2:
            self.xplot_var.set(False)
            self.dualy_var.set(False)

        # Clear existing plot
        self.ax1.clear()
        self.tat_results = {} 
        self.slope_results = {}
                
        # Ensure buttons are reset when Plot is clicked
        self.zoom_x_button.config(state='disabled', text='Zoom X')
        self.zoom_y_button.config(state='disabled', text='Zoom Y')
        self.fft_button.config(state='disabled', text='FFT')
        self.fig2_zoom_x_button.config(state='disabled', text='Zoom X')
        self.fig2_zoom_y_button.config(state='disabled', text='Zoom Y')
        self.first_integral_button.config(state='disabled', text='1st Integral')
        self.second_integral_button.config(state='disabled', text='2nd Integral')

        # Remove existing FFT axes if they exist
        if hasattr(self, 'ax2'):
            self.ax2.clear()
            self.ax2.remove()
            delattr(self, 'ax2')
            self.first_integral_button.config(text='1st Integral')
        if hasattr(self, 'ax2_right'):
            self.ax2_right.clear()
            self.ax2_right.remove()
            delattr(self, 'ax2_right')
            self.second_integral_button.config(text='2nd Integral')
        
        # Calculate max and min values for normalization
        self.max = np.max(self.data[1000:, self.plot_channels], axis=0)
        self.min = np.min(self.data[:, self.plot_channels], axis=0)
        
        if self.xplot_var.get():
            self.plot_cross_plot()
        elif self.dualy_var.get():
            self.plot_double_y()
        else:
            for i, channel in enumerate(self.plot_channels):
                # Normalize data if enabled
                if self.normalize_var.get():
                    if self.max[i] >= abs(self.min[i]):
                        data = self.data[:, channel] / self.max[i]
                    else:
                        data = self.data[:, channel] / abs(self.min[i])
                else:
                    data = self.data[:, channel]

                # Plot data
                self.ax1.tick_params('y', colors='k')
                legend_label = self.hdr[channel][0]
                if self.filter_active:
                    if self.multiplier_active:
                        legend_label += f" ({self.filtfreq[i]:.1f} Hz, {self.magnitude[i]:.1f} Multiplier)"
                    else:  # If multiplier not active, assume it's a single frequency
                        legend_label += f" ({self.filtfreq:.1f} Hz)"
                self.ax1.plot(self.data[:, 0], data, label=legend_label)

            # Set plot labels and legend
            self.ax1.set_xlabel('Time (sec)')
            self.ax1.set_ylabel('Parameter')
            self.ax1.grid(True, alpha=0.2) 
            self.lgnd = self.ax1.legend()

            # Enable buttons
            self.normalize_toggle.config(state='normal')
            self.filter_button.config(state='normal')
            self.filter_magnitude_button.config(state='normal')
            self.fft_button.config(state='normal')
            self.plotly_button.config(state='normal')
            self.zoom_x_button.config(state='normal')
            self.zoom_y_button.config(state='normal')
            self.first_integral_button.config(state='normal')
            self.second_integral_button.config(state='disabled')
            self.save_button.config(state='normal')
            self.xhairs_button.config(state='normal')
            self.plot_button.config(state='normal')
            if not self.normalize_var.get():
                if len(self.plot_channels) != 1:
                    self.slope_button.config(state='disabled')
                else:
                    self.slope_button.config(state='normal')
            if not self.normalize_var.get():
                self.tat_button.config(state='normal')
            if len(self.plot_channels) == 2:
                self.xplot_toggle.config(state='normal')
                self.dualy_toggle.config(state='normal')

        # Update info listbox
        self.update_info_listbox()

        # Update canvas
        self.fig.tight_layout()
        self.canvas.draw()
        self.canvas.flush_events()  # Force update

        # Initialize zoom state
        self.zoom_x_activated = False  # For horizontal zoom
        self.zoom_y_activated = False  # For vertical zoom
        self.fig2_zoom_x_activated = False  # For horizontal zoom
        self.fig2_zoom_y_activated = False  # For vertical zoom
        self.zoom_rect = None
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_move)
        self.canvas.mpl_connect('button_release_event', self.on_release)

    def plot_cross_plot(self):
        self.deactivate_interactions()
        # Clear existing plot
        self.ax1.clear()
        self.tat_results = {} 
        self.slope_results = {}

        # Ensure buttons are reset when Double Y is clicked
        self.fft_button.config(state='disabled')
        self.first_integral_button.config(state='disabled')
        self.slope_button.config(state='disabled')
        self.tat_button.config(state='disabled')
        self.dualy_toggle.config(state='disabled')
        self.xhairs_button.config(state='disabled')
        self.ax2_xhairs_button.config(state='disabled')
        self.plotly_button.config(state='disabled')
        self.plot_button.config(state='disabled')
        self.zoom_x_button.config(state='normal')
        self.zoom_y_button.config(state='normal')

        if len(self.plot_channels) != 2:
            tk.messagebox.showinfo("Cross Plot", "Select exactly two channels.")
            self.xplot_var.set(False)  # Uncheck if not two channels
            return

        channel1_data = self.data[:, self.plot_channels[0]]
        channel2_data = self.data[:, self.plot_channels[1]]

        # Normalize data if enabled
        if self.normalize_var.get():
            channel1_data = (channel1_data - np.min(channel1_data)) / (np.max(channel1_data) - np.min(channel1_data))
            channel2_data = (channel2_data - np.min(channel2_data)) / (np.max(channel2_data) - np.min(channel2_data))

        self.ax1.plot(channel1_data, channel2_data, label="Cross Plot")
        self.ax1.set_xlabel(self.hdr[self.plot_channels[0]][0])
        self.ax1.set_ylabel(self.hdr[self.plot_channels[1]][0])
        self.ax1.grid(True, alpha=0.2) 
        self.ax1.legend()
        self.fig.tight_layout()
        self.canvas.draw()
        self.canvas.flush_events()

    def plot_double_y(self):
        self.deactivate_interactions()
        # Clear existing plot
        self.ax1.clear()
        self.tat_results = {} 
        self.slope_results = {}

        # Ensure buttons are reset when Double Y is clicked
        self.fft_button.config(state='disabled')
        self.first_integral_button.config(state='disabled')
        self.slope_button.config(state='disabled')
        self.tat_button.config(state='disabled')
        self.xplot_toggle.config(state='disabled')
        self.xhairs_button.config(state='disabled')
        self.ax2_xhairs_button.config(state='disabled')
        self.plotly_button.config(state='disabled')
        self.plot_button.config(state='disabled')
        self.zoom_x_button.config(state='normal')

        if len(self.plot_channels) != 2:
            tk.messagebox.showinfo("Double Y", "Select exactly two channels.")
            self.dualy_var.set(False)  # Uncheck if not two channels
            return

        # Create the initial plot with the first channel on the left y-axis
        self.ax1.plot(self.data[:, 0], self.data[:, self.plot_channels[0]], 'b-', label=self.hdr[self.plot_channels[0]][0])
        self.ax1.set_xlabel('Time (sec)')
        self.ax1.set_ylabel(self.hdr[self.plot_channels[0]][0], color='b')
        self.ax1.tick_params('y', colors='b')

        # Create the second y-axis and plot the second channel
        self.ax2 = self.ax1.twinx()
        self.ax2.plot(self.data[:, 0], self.data[:, self.plot_channels[1]], 'r-', label=self.hdr[self.plot_channels[1]][0])
        self.ax2.set_ylabel(self.hdr[self.plot_channels[1]][0], color='r')
        self.ax2.tick_params('y', colors='r')

        # Add a grid
        self.ax1.grid(True, alpha=0.2)

        # Combine legends from both axes
        lines, labels = self.ax1.get_legend_handles_labels()
        lines2, labels2 = self.ax2.get_legend_handles_labels()
        self.ax1.legend(lines + lines2, labels + labels2, loc="upper right")

        # Update canvas
        self.fig.tight_layout()
        self.canvas.draw()
        self.canvas.flush_events()

    def activate_zoom_x(self):
        self.zoom_x_activated = not self.zoom_x_activated
        if self.zoom_x_activated:
            self.active_zoom_axis = 'ax1'  # Set active axis to ax1
            self.zoom_x_button.config(text="Exit Zoom")
            self.zoom_y_activated = False
            self.zoom_y_button.config(text="Zoom Y")
            self.fig2_zoom_x_activated = False
            self.fig2_zoom_y_activated = False
            self.fig2_zoom_x_button.config(text="Zoom X")
            self.fig2_zoom_y_button.config(text="Zoom Y")
        else:
            self.zoom_x_button.config(text="Zoom X")
            self.active_zoom_axis = None

    def activate_zoom_y(self):
        self.zoom_y_activated = not self.zoom_y_activated
        if self.zoom_y_activated:
            self.active_zoom_axis = 'ax1'  # Set active axis to ax1
            self.zoom_y_button.config(text="Exit Zoom")
            self.zoom_x_activated = False
            self.zoom_x_button.config(text="Zoom X")
            self.fig2_zoom_x_activated = False
            self.fig2_zoom_y_activated = False
            self.fig2_zoom_x_button.config(text="Zoom X")
            self.fig2_zoom_y_button.config(text="Zoom Y")
        else:
            self.zoom_y_button.config(text="Zoom Y")
            self.active_zoom_axis = None

    def activate_fig2_zoom_x(self):
        self.fig2_zoom_x_activated = not self.fig2_zoom_x_activated
        if self.fig2_zoom_x_activated:
            self.active_zoom_axis = 'ax2'  # Set active axis to ax2
            self.fig2_zoom_x_button.config(text="Exit Zoom")
            self.zoom_x_activated = False
            self.zoom_y_activated = False
            self.zoom_x_button.config(text="Zoom X")
            self.zoom_y_button.config(text="Zoom Y")
            self.fig2_zoom_y_activated = False
            self.fig2_zoom_y_button.config(text="Zoom Y")
        else:
            self.fig2_zoom_x_button.config(text="Zoom X")
            self.active_zoom_axis = None

    def activate_fig2_zoom_y(self):
        self.fig2_zoom_y_activated = not self.fig2_zoom_y_activated
        if self.fig2_zoom_y_activated:
            self.active_zoom_axis = 'ax2'  # Set active axis to ax2
            self.fig2_zoom_y_button.config(text="Exit Zoom")
            self.zoom_x_activated = False
            self.zoom_y_activated = False
            self.zoom_x_button.config(text="Zoom X")
            self.zoom_y_button.config(text="Zoom Y")
            self.fig2_zoom_x_activated = False
            self.fig2_zoom_x_button.config(text="Zoom X")
        else:
            self.fig2_zoom_y_button.config(text="Zoom Y")
            self.active_zoom_axis = None

    def on_press(self, event):
        if event.button == 1:  # Left mouse button
            if any([self.zoom_x_activated, self.zoom_y_activated, 
                    self.fig2_zoom_x_activated, self.fig2_zoom_y_activated]):
                # Get the current active axis
                current_ax = getattr(self, self.active_zoom_axis) if self.active_zoom_axis else None
                if current_ax:
                    # Start drawing the zoom rectangle
                    self.zoom_rect_start = (event.xdata, event.ydata)
                    self.zoom_rect = plt.Rectangle((event.xdata, event.ydata), 
                                                0, 0,  # Initial width and height are 0
                                                color='gray', alpha=0.2)
                    current_ax.add_patch(self.zoom_rect)

    def on_move(self, event):
        if self.zoom_rect is not None:
            current_ax = getattr(self, self.active_zoom_axis) if self.active_zoom_axis else None
            if current_ax:
                x0, y0 = self.zoom_rect_start
                x1, y1 = event.xdata, event.ydata
                
                if self.zoom_x_activated or self.fig2_zoom_x_activated:
                    self.zoom_rect.set_width(x1 - x0)
                    self.zoom_rect.set_height(current_ax.get_ylim()[1] - current_ax.get_ylim()[0])
                elif self.zoom_y_activated or self.fig2_zoom_y_activated:
                    self.zoom_rect.set_width(current_ax.get_xlim()[1] - current_ax.get_xlim()[0])
                    self.zoom_rect.set_height(y1 - y0)
                
                self.canvas.draw_idle()

    def on_release(self, event):
        if event.button == 1 and self.zoom_rect is not None:
            current_ax = getattr(self, self.active_zoom_axis) if self.active_zoom_axis else None
            if current_ax:
                x0, y0 = self.zoom_rect_start
                x1, y1 = event.xdata, event.ydata

                if x0 > x1:  # Ensure x0 < x1
                    x0, x1 = x1, x0
                if y0 > y1:  # Ensure y0 < y1
                    y0, y1 = y1, y0

                if self.zoom_x_activated or self.fig2_zoom_x_activated:
                    current_ax.set_xlim(x0, x1)
                elif self.zoom_y_activated or self.fig2_zoom_y_activated:
                    current_ax.set_ylim(y0, y1)

                self.zoom_rect.remove()
                self.zoom_rect = None
                self.canvas.draw_idle()

    def deactivate_interactions(self):
        """Deactivates all x-hairs and zoom states and resets buttons."""
        if self.xhairs_button.cget('text') == 'Exit X-hairs':
            self.toggle_xhairs()  # Toggle off Fig1 X-hairs
        if self.ax2_xhairs_button.cget('text') == 'Exit X-hairs':
            self.toggle_ax2_xhairs()  # Toggle off Fig2 X-hairs

        self.zoom_x_activated = False
        self.zoom_y_activated = False
        self.fig2_zoom_x_activated = False
        self.fig2_zoom_y_activated = False

        self.zoom_x_button.config(text='Zoom X')
        self.zoom_y_button.config(text='Zoom Y')
        self.fig2_zoom_x_button.config(text='Zoom X')
        self.fig2_zoom_y_button.config(text='Zoom Y')

        self.active_zoom_axis = None
        if self.zoom_rect is not None:
            self.zoom_rect.remove()
            self.zoom_rect = None
            self.canvas.draw_idle()

    def update_info_listbox(self):
        # Update info listbox with channel information
        self.info_listbox.delete(0, 'end')
        for i, channel in enumerate(self.plot_channels):
            # Get TAT info if it exists
            tat_info = self.tat_results.get(channel, {'threshold': '', 'tat': ''})
            threshold_str = f"{tat_info['threshold']:.4f}" if tat_info['threshold'] != '' else ''
            tat_str = f"{tat_info['tat']:.6f}" if tat_info['tat'] != '' else ''
            
            # Get slope info if it exists
            slope_info = getattr(self, 'slope_results', {}).get(channel, {'slope': '', 'run': '', 'rise': ''})
            slope_str = f"{slope_info['slope']:.0f}" if slope_info['slope'] != '' else ''
            run_str = f"{slope_info['run']:.3f}" if slope_info['run'] != '' else ''
            rise_str = f"{slope_info['rise']:.3f}" if slope_info['rise'] != '' else ''
            
            # Build the display string based on what information exists
            base_info = f"{self.hdr[channel][0]:<10} {self.max[i]:>15.3f} {self.min[i]:>15.3f}"
            
            if threshold_str or slope_str:  
                if threshold_str and slope_str:
                    full_info = f"{base_info} {threshold_str:>15} {tat_str:>20} {slope_str:>15} {run_str:>20} {rise_str:>17}"
                elif threshold_str and not slope_str:
                    full_info = f"{base_info} {threshold_str:>15} {tat_str:>20}"
                elif not threshold_str and slope_str:
                    full_info = f"{base_info} {slope_str:>65} {run_str:>15} {rise_str:>20}"
            else:  # If no TAT info, just show slope info
                full_info = f"{base_info}"
                
            self.info_listbox.insert('end', full_info)

    def apply_filter(self):
        # Check if any channels are selected
        self.restore_channel_selections()
        self.deactivate_interactions()

        # Get filter frequency from entry
        try:
            self.filtfreq = float(self.filter_entry.get())
        except ValueError:
            tk.messagebox.showerror("Invalid Input", "Please enter a valid filter frequency.")
            return

        # Calculate filter coefficients
        nyquist = (1 / (self.data[1, 0] - self.data[0, 0])) / 2
        Wn = self.filtfreq / nyquist

        if Wn >= 1:
            Wn = 0.95
            self.filter_entry.delete(0, 'end')
            self.filter_entry.insert(0, str(Wn * nyquist))
        # Create filter
        self.b, self.a = butter(2, Wn)

        # Store original data if not already stored
        if not hasattr(self, 'original_data'):
            self.original_data = self.data.copy()
        else:
            # Reset to original data before applying new filter
            self.data = self.original_data.copy()

        # Apply filter to each data column (excluding time column)
        for col in range(1, self.data.shape[1]):
            self.data[:, col] = filtfilt(self.b, self.a, self.data[:, col])

        # Force the listbox to update its display
        self.channel_listbox.update()
        self.multiplier_active = False
        self.filter_active = True
        self.reset_filter_button.config(state='normal')

        # Replot data with filter
        self.plot_data()

    def open_filter_magnitude_window(self):
        self.restore_channel_selections()
        self.deactivate_interactions()

        self.filter_magnitude_window = tk.Toplevel(self.master)
        self.filter_magnitude_window.title("Filter & Multiplier")

        # Create table headers and entry boxes within a grid layout
        table_frame = tk.Frame(self.filter_magnitude_window)
        table_frame.pack(pady=5)

        filter_header = tk.Label(table_frame, text="Filter (Hz)", width=15)
        filter_header.grid(row=0, column=1)  # Header in row 0, column 1

        magnitude_header = tk.Label(table_frame, text="Multiplier", width=15)
        magnitude_header.grid(row=0, column=2)  # Header in row 0, column 2

        # Create Entry boxes for each channel in a table-like format
        self.filter_entries = {}
        self.magnitude_entries = {}
        for i, channel in enumerate(self.plot_channels):
            channel_label = tk.Label(table_frame, text=f"{self.hdr[channel][0]}:")
            channel_label.grid(row=i+1, column=0, sticky="w") # Channel label in column 0

            filter_entry = ttk.Entry(table_frame, width=10)
            filter_entry.insert(0, "100")  # Default value
            filter_entry.grid(row=i+1, column=1, padx=1)  # Entry in column 1
            self.filter_entries[channel] = filter_entry

            magnitude_entry = ttk.Entry(table_frame, width=10)
            magnitude_entry.insert(0, "1")  # Default magnitude
            magnitude_entry.grid(row=i+1, column=2, padx=1)  # Entry in column 2
            self.magnitude_entries[channel] = magnitude_entry

        # Apply and Cancel Buttons
        button_frame = tk.Frame(self.filter_magnitude_window)
        button_frame.pack(pady=10)

        apply_button = ttk.Button(button_frame, text="Apply", command=self.apply_filter_magnitude)
        apply_button.pack(side='left', padx=5)

        cancel_button = ttk.Button(button_frame, text="Cancel", command=self.filter_magnitude_window.destroy)
        cancel_button.pack(side='left', padx=5)

    def apply_filter_magnitude(self):
        self.restore_channel_selections()
        self.deactivate_interactions()

        try:   
            # Store original data if not already stored
            if not hasattr(self, 'original_data'):
                self.original_data = self.data.copy()
            else:
                # Reset to original data before applying new filter
                self.data = self.original_data.copy()

            self.filtfreq = []
            self.magnitude = []

            for channel, entry in self.filter_entries.items():
                try:
                    self.filtfreq.append(float(entry.get()))
                    self.magnitude.append(float(self.magnitude_entries[channel].get())) # Get magnitude for this channel 
                except ValueError:
                    tk.messagebox.showerror("Invalid Input", f"Please enter valid values for {self.hdr[channel][0]}.")
                    return

                # Calculate filter coefficients
                nyquist = (1 / (self.data[1, 0] - self.data[0, 0])) / 2
                Wn = self.filtfreq[-1] / nyquist # Use the last added filter frequency

                if Wn >= 1:
                    Wn = 0.95
                    entry.delete(0, 'end')
                    entry.insert(0, str(Wn * nyquist))
                    self.filtfreq[-1] = Wn * nyquist

                # Create filter
                self.b, self.a = butter(2, Wn)

                # Apply filter and magnitude scaling
                self.data[:, channel] = filtfilt(self.b, self.a, self.data[:, channel]) * self.magnitude[-1]
                self.multiplier_active = True
                self.filter_active = True
                self.reset_filter_button.config(state='normal')

        except ValueError:
            tk.messagebox.showerror("Invalid Input", "Please enter valid filter frequency and magnitude values.")
            return

        self.filter_magnitude_window.destroy()
        self.plot_data()  # Replot the data

    def reset_filter(self):
        self.restore_channel_selections()
        self.deactivate_interactions()

        """Reset data to unfiltered state"""
        if hasattr(self, 'original_data'):
            self.data = self.original_data.copy()
            self.b = np.array([])
            self.a = np.array([])
            self.filter_active = False
            self.reset_filter_button.config(state='disabled')
            self.plot_data()

    def toggle_fft(self):
        self.restore_channel_selections()
        self.deactivate_interactions()

        if hasattr(self, 'ax2'):  # FFT is currently displayed
            # Store current ax1 data and properties
            lines_data = []
            for line in self.ax1.lines:
                lines_data.append({
                    'xdata': line.get_xdata(),
                    'ydata': line.get_ydata(),
                    'color': line.get_color(),
                    'label': line.get_label()
                })
            xlim = self.ax1.get_xlim()
            ylim = self.ax1.get_ylim()

            # Remove both axes
            self.ax1.remove()
            self.ax2.remove()
            delattr(self, 'ax2')

            # Create new single axis
            self.ax1 = self.fig.add_subplot(111)

            # Replot the data
            for line_data in lines_data:
                self.ax1.plot(line_data['xdata'], line_data['ydata'], 
                            color=line_data['color'], 
                            label=line_data['label'])

            # Restore axis properties
            self.ax1.set_xlim(xlim)
            self.ax1.set_ylim(ylim)
            self.ax1.set_xlabel('Time (sec)')
            self.ax1.set_ylabel('Parameter')
            self.ax1.grid(True, alpha=0.2)
            self.ax1.legend()

            # Update layout and redraw
            self.fig.tight_layout()
            self.canvas.draw()

            # Update button states
            self.fft_button.config(text="FFT")
            self.fig2_zoom_x_button.config(state='disabled', text='Zoom X')
            self.fig2_zoom_y_button.config(state='disabled', text='Zoom Y')
            self.first_integral_button.config(state='normal')
            self.plot_button.config(state='normal')
            self.ax2_xhairs_button.config(state='disabled', text='X-hairs')
            self.test_combobox.config(state='normal')
            self.channel_listbox.config(state='normal')
        else:
            # Calculate and display FFT
            self.calculate_fft()
            self.first_integral_button.config(state='disabled')
            self.second_integral_button.config(state='disabled')
            self.plot_button.config(state='disabled')
            self.ax2_xhairs_button.config(state='normal')
            self.test_combobox.config(state='disabled')
            self.channel_listbox.config(state='disabled')
            self.fig2_zoom_x_button.config(state='normal')
            self.fig2_zoom_y_button.config(state='normal')

    def calculate_fft(self):
        # Store current ax1 limits and properties before removing it
        current_xlim = self.ax1.get_xlim()
        current_ylim = self.ax1.get_ylim()
        lines_data = []
        for line in self.ax1.lines:
            lines_data.append({
                'xdata': line.get_xdata(),
                'ydata': line.get_ydata(),
                'color': line.get_color(),
                'label': line.get_label()
            })

        # Get x-axis limits for FFT calculation
        xlim = self.ax1.get_xlim()
        tmp = np.where((self.data[:, 0] >= xlim[0]) & (self.data[:, 0] <= xlim[1]))[0]

        fs = 1 / (self.data[1, 0] - self.data[0, 0])

        # Initialize freq and mag with an initial size
        self.freq = np.zeros((1, len(self.plot_channels)))  
        self.mag = np.zeros((1, len(self.plot_channels))) 

        for i, channel in enumerate(self.plot_channels):
            n = len(self.data[tmp, channel])
            nfft = 2**(int(np.log2(n))) 
            if nfft != n:
                nfft = n

            freq_temp, mag_temp = self.calculate_fft_freq_mag(self.data[tmp, channel], fs)

            if nfft // 2 > self.freq.shape[0]:
                self.freq.resize((nfft // 2, len(self.plot_channels)), refcheck=False)
                self.mag.resize((nfft // 2, len(self.plot_channels)), refcheck=False)

            self.freq[:, i] = freq_temp 
            self.mag[:, i] = mag_temp

        # Create new subplot arrangement
        self.ax1.remove()  # Remove the original single axes
        self.ax1 = self.fig.add_subplot(121)  # Time domain plot
        self.ax2 = self.fig.add_subplot(122)  # FFT plot

        # Replot time domain data with stored properties
        for line_data in lines_data:
            self.ax1.plot(line_data['xdata'], line_data['ydata'], 
                        color=line_data['color'], 
                        label=line_data['label'])

        # Restore ax1 properties
        self.ax1.set_xlim(current_xlim)
        self.ax1.set_ylim(current_ylim)
        self.ax1.set_xlabel('Time (sec)')
        self.ax1.set_ylabel('Parameter')
        self.ax1.grid(True, alpha=0.2) 
        self.ax1.legend()

        # Plot FFT data
        for i, channel in enumerate(self.plot_channels):
            self.ax2.plot(self.freq[:, i], self.mag[:, i], label=self.hdr[channel][0])

        self.ax2.set_xscale('log')
        ymin = np.min(self.mag)
        ymax = np.max(self.mag)
        padding = 0.05 * (ymax - ymin)
        self.ax2.set_ylim(ymin - padding, ymax + padding) # Or set ymin to 0 for a base at zero
        # self.ax2.relim() # Recalculate data limits
        # self.ax2.autoscale(enable=True, axis='y', tight=True) # Autoscale the view
        self.ax2.set_xlabel('Freq (Hz)')
        self.ax2.set_ylabel('Power $(G^2_{RMS})$')
        self.ax2.grid(True, which='both', alpha=0.2)
        self.ax2.legend()

        # Adjust layout and redraw
        self.fig.tight_layout()
        self.canvas.draw()

        # Update button states
        self.fft_button.config(text="Exit FFT")

    def calculate_fft_freq_mag(self, signal, fs):
        """
        Calculate power spectral density using FFT.

        Args:
            signal (array-like): Input signal.
            fs (float): Sampling frequency of the signal.

        Returns:
            tuple: Frequency vector and corresponding PSD values.
        """

        n = len(signal)
        nfft = 2**(int(np.log2(n))) 
        if nfft != n:
            nfft = n

        # Calculate FFT
        fftx = np.fft.fft(signal, nfft)
        
        mx = 2 * (np.abs(fftx[1:(nfft//2)+1])) / n

        # Calculate RMS magnitude, power, and dB
        mxRMS = np.sqrt(2) / 2 * mx
        mxPWR = mxRMS**2
        mxdB = 10 * np.log10(mxPWR)
        
        # Frequency vector
        freq = (fs * np.arange(0, nfft // 2) / nfft)

        return freq, mxPWR

    def toggle_first_integral(self):
        self.restore_channel_selections()
        self.deactivate_interactions()

        if hasattr(self, 'ax2'):  # Integral plot is currently displayed
            if hasattr(self, 'ax2_right'):  # If right y-axis exists, we're exiting
                # Remove the right y-axis and its plots
                self.ax2_right.remove()
                delattr(self, 'ax2_right')
            # Store current ax1 data and properties
            lines_data = []
            for line in self.ax1.lines:
                lines_data.append({
                    'xdata': line.get_xdata(),
                    'ydata': line.get_ydata(),
                    'color': line.get_color(),
                    'label': line.get_label()
                })
            xlim = self.ax1.get_xlim()
            ylim = self.ax1.get_ylim()

            # Remove both axes
            self.ax1.remove()
            self.ax2.remove()
            delattr(self, 'ax2')

            # Create new single axis
            self.ax1 = self.fig.add_subplot(111)

            # Replot the data
            for line_data in lines_data:
                self.ax1.plot(line_data['xdata'], line_data['ydata'], 
                            color=line_data['color'], 
                            label=line_data['label'])

            # Restore axis properties
            self.ax1.set_xlim(xlim)
            self.ax1.set_ylim(ylim)
            self.ax1.set_xlabel('Time (sec)')
            self.ax1.set_ylabel('Parameter')
            self.ax1.grid(True, alpha=0.2)
            self.ax1.legend()

            # Update layout and redraw
            self.fig.tight_layout()
            self.canvas.draw()

            # Update button states
            self.first_integral_button.config(text='1st Integral')
            self.second_integral_button.config(text='2nd Integral')
            self.second_integral_button.config(state=tk.DISABLED)
            self.fft_button.config(state='normal')
            self.mps_radio.config(state='normal')
            self.fps_radio.config(state='normal')
            self.plot_button.config(state='normal')
            self.ax2_xhairs_button.config(state='disabled', text='X-hairs')
            self.test_combobox.config(state='normal')
            self.channel_listbox.config(state='normal')
            self.fig2_zoom_x_button.config(state='disabled', text='Zoom X')
            self.fig2_zoom_y_button.config(state='disabled', text='Zoom Y')
        else:
            self.calculate_first_integral()   
            self.second_integral_button.config(state='normal')  
            self.plot_button.config(state='disabled')
            self.ax2_xhairs_button.config(state='normal')
            self.test_combobox.config(state='disabled')
            self.channel_listbox.config(state='disabled')
            self.fig2_zoom_x_button.config(state='normal')
            self.fig2_zoom_y_button.config(state='normal')

    def calculate_first_integral(self):
        self.deactivate_interactions()
        # Get initial condition value and Decel checkbox state
        try:
            initial_condition = float(self.initial_condition_entry.get())
        except ValueError:
            tk.messagebox.showerror("Invalid Input", "Please enter a valid number for the initial condition.")
            return

        # Store current ax1 limits and properties before removing it
        current_xlim = self.ax1.get_xlim()
        current_ylim = self.ax1.get_ylim()
        lines_data = []
        for line in self.ax1.lines:
            lines_data.append({
                'xdata': line.get_xdata(),
                'ydata': line.get_ydata(),
                'color': line.get_color(),
                'label': line.get_label()
            })

        # Get x-axis limits for integral calculation
        xlim = self.ax1.get_xlim()
        tmp = np.where((self.data[:, 0] >= xlim[0]) & (self.data[:, 0] <= xlim[1]))[0]

        # Calculate time vector and initialize integral array
        self.t_integ = self.data[tmp, 0]
        self.frst_integ = np.zeros((len(tmp), len(self.plot_channels)))

        # Calculate integral for each channel
        for i, channel in enumerate(self.plot_channels):
            data = self.data[tmp, channel]
            
            # Convert to fps if acceleration
            try:
                if self.hdr[channel][1].lower() == 'g':
                    if self.integral_type.get() == 'ft/s':
                        data *= 32.1740486 # g to ft/s^2
                    else:
                        data *= 9.80665 # g to m/s^2
            except IndexError:
                if any(substring in self.hdr[channel][0].lower() for substring in ('ax', 'ay', 'az')): 
                    if self.integral_type.get() == 'ft/s':
                        data *= 32.1740486 # g to ft/s^2
                    else:
                        data *= 9.80665 # g to m/s^2

            # Invert integral if Decel is checked
            if self.decel_var.get():
                self.frst_integ[:, i] = initial_condition - cumulative_trapezoid(data, self.t_integ, initial=0)
            else:
                self.frst_integ[:, i] = cumulative_trapezoid(data, self.t_integ, initial=0) + initial_condition

        # Create new subplot arrangement
        self.ax1.remove()  # Remove the original single axes
        self.ax1 = self.fig.add_subplot(121)  # Time domain plot
        self.ax2 = self.fig.add_subplot(122)  # 1st Integral plot

        # Replot time domain data with stored properties
        for line_data in lines_data:
            self.ax1.plot(line_data['xdata'], line_data['ydata'], 
                        color=line_data['color'], 
                        label=line_data['label'])

        # Restore ax1 properties
        self.ax1.set_xlim(current_xlim)
        self.ax1.set_ylim(current_ylim)
        self.ax1.set_xlabel('Time (sec)')
        self.ax1.set_ylabel('Parameter')
        self.ax1.grid(True, alpha=0.2) 
        self.ax1.legend()

        # Plot 1st Integral data
        for i, channel in enumerate(self.plot_channels):
                self.ax2.plot(self.t_integ, self.frst_integ[:, i], label=self.hdr[channel][0])
        self.ax2.set_xlabel('Time (sec)')
        self.ax2.set_xlim(current_xlim)
        try:
            if self.hdr[channel][1].lower() == 'g':
                if self.integral_type.get() == 'ft/s':
                    self.ax2.set_ylabel('Velocity (ft/s)')
                else:
                    self.ax2.set_ylabel('Velocity (m/s)')
            else:
                self.ax2.set_ylabel('Parameter')
        except IndexError:
            if any(substring in self.hdr[channel][0].lower() for substring in ('ax', 'ay', 'az')):
                if self.integral_type.get() == 'ft/s':
                    self.ax2.set_ylabel('Velocity (ft/s)')
                else:
                    self.ax2.set_ylabel('Velocity (m/s)')
            else:
                self.ax2.set_ylabel('Parameter')

        self.ax2.grid(True, alpha=0.2)
        self.ax2.legend()
    
        # Update layout and redraw
        self.fig.tight_layout()
        self.canvas.draw()

        # Update button states
        self.first_integral_button.config(text="Exit Integral")
        self.second_integral_button.config(state='normal')
        self.fft_button.config(state='disabled')
        self.mps_radio.config(state='disabled')
        self.fps_radio.config(state='disabled')

    def calculate_second_integral(self):
        self.deactivate_interactions()
        self.restore_channel_selections()

        if hasattr(self, 'ax2_right'):  # If right y-axis exists, we're exiting
            # Remove the right y-axis and its plots
            self.ax2_right.remove()
            delattr(self, 'ax2_right')
            
            # Update button text
            self.second_integral_button.config(text='2nd Integral')
            self.ax2_xhairs_button.config(state='normal')
            
            # Adjust layout to accommodate the new labels
            self.fig.tight_layout() 

            # Redraw canvas
            self.canvas.draw()
        else:
            # Calculate second integral (displacement) for selected channels
            if hasattr(self, 'ax2'):
                # Calculate integral
                self.scnd_integ = np.zeros((len(self.t_integ), len(self.plot_channels)))
                for i, channel in enumerate(self.plot_channels):
                    self.scnd_integ[:, i] = cumulative_trapezoid(self.frst_integ[:, i], self.t_integ, initial=0)
                    if self.integral_type.get() == 'ft/s':
                        self.scnd_integ[:, i] *= 12 # ft to inch
                    
                # Create right y-axis on ax2
                self.ax2_right = self.ax2.twinx()
                
                # Plot second integral on right y-axis
                for i, channel in enumerate(self.plot_channels):
                    self.ax2_right.plot(self.t_integ, self.scnd_integ[:, i], '--', 
                                    label=f"{self.hdr[channel][0]} (2nd Int)")
                
                # Set y-axis label and legend
                try:
                    if self.hdr[channel][1].lower() == 'g':
                        if self.integral_type.get() == 'ft/s':
                            self.ax2_right.set_ylabel('Displacement (in)')
                        else:
                            self.ax2_right.set_ylabel('Displacement (m)')
                    else:
                        self.ax2_right.set_ylabel('Parameter')
                except IndexError:
                    if any(substring in self.hdr[channel][0].lower() for substring in ('ax', 'ay', 'az')):
                        if self.integral_type.get() == 'ft/s':
                            self.ax2_right.set_ylabel('Displacement (in)')
                        else:
                            self.ax2_right.set_ylabel('Displacement (m)')
                    else:
                        self.ax2_right.set_ylabel('Parameter')
                

                # Update button text
                self.second_integral_button.config(text='Exit Integral')
                self.ax2_xhairs_button.config(state='disabled')

                # Adjust layout to accommodate the new labels
                self.fig.tight_layout() 

                # Redraw canvas
                self.canvas.draw()       

    def plot_to_plotly(self):
        self.restore_channel_selections()

        fig = go.Figure()

        for i, channel in enumerate(self.plot_channels):
            # Normalize data if enabled
            if self.normalize_var.get():
                if self.max[i] >= abs(self.min[i]):
                    data = self.data[:, channel] / self.max[i]
                else:
                    data = self.data[:, channel] / abs(self.min[i])
            else:
                data = self.data[:, channel]

            fig.add_trace(go.Scatter(
                x=self.data[:, 0], 
                y=data,
                mode='lines',
                name=self.hdr[channel][0]  
            ))

        fig.update_layout(
            xaxis_title="Time (sec)",
            yaxis_title="Parameter",
            title="DAS Quick Look"
        )
        fig.show()

    def toggle_xhairs(self):
        self.restore_channel_selections()

        if self.xhairs_button.cget('text') == 'Exit X-hairs':
            self.canvas.mpl_disconnect(self.motion_cid)
            self.xhairs_button.config(text='X-hairs')
            self.cursor.disconnect()
            del self.cursor
            self.canvas.draw()
            return
        else:
            self.zoom_x_activated = False
            self.zoom_y_activated = False
            self.zoom_x_button.config(text='Zoom X')
            self.zoom_y_button.config(text='Zoom Y')
            self.xhairs_button.config(text='Exit X-hairs')
            self.cursor = SnaptoCursor(self.ax1, self.data[:, 0], [self.data[:, i] for i in self.plot_channels], self.hdr, self.plot_channels, self.fft_button.cget('text'))
            self.motion_cid = self.canvas.mpl_connect("motion_notify_event", self.cursor.on_mouse_move)

    def toggle_ax2_xhairs(self):
        if hasattr(self, 'ax2'): 
            if self.ax2_xhairs_button.cget('text') == 'Exit X-hairs':
                self.canvas.mpl_disconnect(self.motion_cid)
                self.ax2_xhairs_button.config(text='X-hairs')
                self.cursor.disconnect()
                del self.cursor
                self.canvas.draw()
                return
            else:
                self.fig2_zoom_x_activated = False
                self.fig2_zoom_y_activated = False
                self.fig2_zoom_x_button.config(text='Zoom X')
                self.fig2_zoom_y_button.config(text='Zoom Y')
                self.ax2_xhairs_button.config(text='Exit X-hairs')

                # Determine the correct x and y data based on active plots
                if self.second_integral_button.cget('text') == 'Exit Integral':
                    # Include both 1st and 2nd integral data for SnaptoCursor
                    x_data = self.t_integ
                    y_data = [self.frst_integ[:, i] for i in range(len(self.plot_channels))] + \
                            [self.scnd_integ[:, i] for i in range(len(self.plot_channels))]

                    # Update headers for both integrals
                    headers = self.hdr + [f"{h[0]} (2nd Int)" for h in self.hdr] 
                    channel_indices = list(range(len(self.hdr))) + list(range(len(self.hdr)))

                elif self.first_integral_button.cget('text') == 'Exit Integral':
                    x_data = self.t_integ
                    y_data = [self.frst_integ[:, i] for i in range(len(self.plot_channels))]
                    headers = self.hdr
                    channel_indices = self.plot_channels
                else:  # Assume FFT plot is active
                    x_data = self.freq[:, 0]
                    y_data = [self.mag[:, i] for i in range(len(self.plot_channels))]
                    headers = self.hdr
                    channel_indices = self.plot_channels

                self.cursor = SnaptoCursor(self.ax2, x_data, y_data, headers, channel_indices, self.fft_button.cget('text')) 
                self.motion_cid = self.canvas.mpl_connect("motion_notify_event", self.cursor.on_mouse_move)
        else:
            tk.messagebox.showwarning("No 2nd Figure", "Generate a 2nd figure first.")

    def calculate_slope(self):
        self.restore_channel_selections()
        self.deactivate_interactions()

        # If slope is already shown, remove it
        if self.slope_button.cget('text') == 'Remove Slope':
            # Remove the slope points and line
            for line in self.ax1.lines[:]:  # Make a copy of the list to iterate over
                if line.get_marker() == 'o' or line.get_linestyle() == '--':
                    line.remove()
            
            # Update canvas
            self.canvas.draw()
            self.update_info_listbox()
            self.canvas.draw_idle()
            
            # Reset button text
            self.slope_button.config(text='2-pt Slope')
            return

        self.cursor = SnaptoCursor(self.ax1, self.data[:, 0], [self.data[:, i] for i in self.plot_channels], self.hdr, self.plot_channels)
        self.motion_cid = self.canvas.mpl_connect("motion_notify_event", self.cursor.on_mouse_move)

        # Create click event handler
        def onclick(event):
            if event.inaxes != self.ax1:
                return
            
            # Get the cursor's current position
            x = self.cursor.ly.get_xdata()[0]  # Get x position from vertical line
            y = self.cursor.lx.get_ydata()[0]  # Get y position from horizontal line

            # Add point
            self.points.append((x, y))
            
            # Draw point on plot
            self.ax1.plot(x, y, 'ro')
            self.canvas.draw()

            # If we have two points, calculate slope
            if len(self.points) == 2:
                # Disconnect click event
                self.canvas.mpl_disconnect(self.click_cid)
                self.canvas.mpl_disconnect(self.motion_cid)
                
                # Disconnect cursor
                self.cursor.disconnect()
                del self.cursor
                self.canvas.draw()

                # Calculate slope
                run = self.points[1][0] - self.points[0][0]
                rise = self.points[1][1] - self.points[0][1]
                slope = rise / run

                # Store slope results
                self.slope_results = {
                    self.plot_channels[0]: {  # Currently using first channel
                        'slope': slope,
                        'run': run,
                        'rise': rise
                    }
                }

                # Draw line between points
                self.ax1.plot([self.points[0][0], self.points[1][0]], [self.points[0][1], self.points[1][1]], 'r--')
                
                # Update info listbox
                self.info_listbox.insert('end', '')
                self.info_listbox.insert('end', 
                    f"2-pt slope: {slope:.0f}, dx = {run:.3f} sec, dy = {rise:.3f} chn_units")
                
                # Update canvas
                self.canvas.draw()
                
                # Reset points for next calculation
                self.points = []

                # Change button text
                self.slope_button.config(text='Remove Slope')
                self.update_info_listbox()

        # Connect click event
        self.click_cid = self.canvas.mpl_connect('button_press_event', onclick)
        
        # Show instructions
        tk.messagebox.showinfo("Slope Calculation", "Click two points on the plot to calculate slope.\nFirst point will be start, second point will be end.")

    def calculate_tat(self):
        """Calculate Time Above Threshold (TAT) for signal data"""
        # Check if any channels are selected
        self.restore_channel_selections()
        self.deactivate_interactions()

        while True:
            try:
                threshold_input = tk.simpledialog.askstring("TAT", "Enter threshold level:")
                # Handle case where user clicks Cancel or closes dialog
                if threshold_input is None:
                    return  # Exit the method if user cancels
                threshold = float(threshold_input)
                break  # Exit the loop if conversion to float is successful

            except ValueError:
                tk.messagebox.showerror("Invalid Input", "Please enter a valid threshold level.")

        # Get absolute value option from user input
        absolute = tk.messagebox.askquestion("TAT", "Use absolute value of signal?")

        # Calculate time step
        dt = self.data[1, 0] - self.data[0, 0]
        num_channels = len(self.plot_channels)

        # Store TAT results
        self.tat_results = {}  # Dictionary to store threshold and TAT values

        # Process each channel
        for jj in range(num_channels):
            invert = 1 if threshold >= 0 else -1

            # Check if data is being displayed filtered
            # if not self.filter_active:
            channel_data = self.data[:, self.plot_channels[jj]]
            if absolute == 'yes':
                v_gte_thresh = np.where(np.abs(channel_data) >= invert * threshold)[0]
            else:
                v_gte_thresh = np.where(invert * channel_data >= invert * threshold)[0]

            # Calculate duration above threshold
            t_gte_thresh = len(v_gte_thresh) * dt

            # Store the results
            self.tat_results[self.plot_channels[jj]] = {
                'threshold': threshold,
                'tat': t_gte_thresh}

            # Update the info listbox with TAT information
            self.update_info_listbox()    

    def save_current_figure(self):
        # Prompt user for filename
        figpath = filedialog.askdirectory()
        figname = tk.simpledialog.askstring("Save Figure", "Enter filename for figure:")
        if not figname:  # User cancelled or entered nothing
            return

        # Create full file path
        fname = os.path.join(figpath, f"{figname}.tif")
        
        try:
            if hasattr(self, 'ax2'):  # FFT is currently displayed
                # Save both plots
                self.fig.savefig(fname, format='tif')
            else:
                # Create a new figure for single plot
                fig, ax = plt.subplots()

                # Copy the legend and plot data to the new figure
                if self.lgnd is not None:
                    handles, labels = self.ax1.get_legend_handles_labels()
                    ax.legend(handles, labels) 
                for line in self.ax1.lines:
                    ax.plot(line.get_xdata(), line.get_ydata(), 
                            color=line.get_color(), 
                            linestyle=line.get_linestyle(), 
                            marker=line.get_marker())

                # Set the same labels and limits
                ax.set_xlabel(self.ax1.get_xlabel())
                ax.set_ylabel(self.ax1.get_ylabel())
                ax.set_xlim(self.ax1.get_xlim())
                ax.set_ylim(self.ax1.get_ylim())
                ax.grid(True, which='both', alpha=0.2)

                # Save the figure
                fig.savefig(fname, format='tif')
                plt.close(fig)  # Close the temporary figure

        except Exception as e:
            tk.messagebox.showerror("Error", f"Could not save figure: {e}")

    def import_dts(self, filepath):
        """
        Imports data from DTS SliceWare binary files.

        Args:
            dts_pn (str): Path to the directory containing the .dts and .chn files.

        Returns:
            tuple: A tuple containing the data matrix and header information.
        """

        # Find .dts file for header information
        # Get directory path
        dts_dir = os.path.dirname(filepath) + os.sep        

        dts_files = [f for f in os.listdir(dts_dir) if f.endswith('.dts')]
        if not dts_files:
            raise FileNotFoundError("No .dts file found in the directory.")
        dts_file = os.path.join(dts_dir, dts_files[0])

        # Parse .dts file
        dts_info = []
        dts_mod = []
        with open(dts_file, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            if "x.m.l." in first_line.lower():  # Case-insensitive check
                f.close()
                f = open(dts_file, 'r', encoding='utf-16-le')          
            for line in f:
                if '<AnalogInputChanel' in line:  
                    element = ET.fromstring(line)
                    channel_data = {}
                    for attribute, value in element.attrib.items():
                        channel_data[attribute] = value
                    dts_info.append(channel_data)
                elif '<Module ' in line:  # Check for Module lines
                    module_data = {}
                    for match in re.finditer(r'(\w+)="([^"]*)"', line):
                        key = match.group(1)
                        value = match.group(2)
                        module_data[key] = value
                    dts_mod.append(module_data)  # Append to dts_mod


        # Extract channel descriptions and serial numbers
        description = []
        sn = []
        for entry in dts_info:
            for key, value in entry.items():  # Iterate through key-value pairs
                if 'Description' in key:
                    description.append(value)
                elif 'Serial' in key:
                    sn.append(value)
        chan_info = [f"{desc} - {s}" for desc, s in zip(description, sn)]

        # Find .chn files
        chn_files = [f for f in os.listdir(dts_dir) if f.endswith('.chn')]
        if not chn_files:
            raise FileNotFoundError("No .chn files found in the directory.")

        # Organize filenames sequentially
        fnames = sorted(chn_files, key=lambda x: int(x.split('.')[-2]))

        mod_hdr = {}
        for mod_data in dts_mod:
            for key, value in mod_data.items():  # Iterate through key-value pairs
                mod_hdr[key.strip('<').strip('>').replace(' ', '_')] = value.strip('"')
        start_record_sample_number = int(mod_hdr['StartRecordSampleNumber'])

        # Initialize data and header arrays       
        data = None
        hdr = [['Time']]  
        # Import data from .chn files
        for i, filename in enumerate(fnames):
            with open(os.path.join(dts_dir, filename), 'rb') as f:
                # Read header information
                HDR = {}
                HDR['MagicID'] = int(np.fromfile(f, dtype=np.uint32, count=1)[0])
                HDR['version'] = int(np.fromfile(f, dtype=np.uint32, count=1)[0])

                # Handle different versions
                if HDR['version'] in (1,2,3,4):
                    HDR['OffsetToData'] = int(np.fromfile(f, dtype=np.uint64, count=1)[0])
                    HDR['nSamples'] = int(np.fromfile(f, dtype=np.uint64, count=1)[0])
                    HDR['nBits'] = int(np.fromfile(f, dtype=np.uint32, count=1)[0])
                    HDR['SignUnsign'] = int(np.fromfile(f, dtype=np.uint32, count=1)[0])
                    HDR['SampleRate'] = float(np.fromfile(f, dtype=np.double, count=1)[0])
                    HDR['nTriggers'] = int(np.fromfile(f, dtype=np.uint16, count=1)[0])
                    HDR['TriggerSampleNumber'] = int(np.fromfile(f, dtype=np.uint64, count=1)[0])
                    HDR['prezero_cnts'] = int(np.fromfile(f, dtype=np.int32, count=1)[0])
                    if HDR['version'] in (2,3,4):
                        HDR['ADC_Removed_cnts'] = int(np.fromfile(f, dtype=np.int32, count=1)[0])
                    HDR['precal_cnts'] = int(np.fromfile(f, dtype=np.int32, count=1)[0])
                    HDR['snr_percentFS'] = float(np.fromfile(f, dtype=np.double, count=1)[0])
                    HDR['postzero_cnts'] = int(np.fromfile(f, dtype=np.int32, count=1)[0])
                    HDR['postcal_cnts'] = int(np.fromfile(f, dtype=np.int32, count=1)[0])
                    HDR['datazero_cnts'] = int(np.fromfile(f, dtype=np.int32, count=1)[0])
                    HDR['mv_per_cnt'] = float(np.fromfile(f, dtype=np.double, count=1)[0])
                    HDR['engU_per_mV'] = float(np.fromfile(f, dtype=np.double, count=1)[0])
                    HDR['nBytes'] = int(np.fromfile(f, dtype=np.uint16, count=1)[0])
                    HDR['EngUnit'] = str(f.read(HDR['nBytes'] - 1).decode('utf-8')).rstrip()
                    if HDR['version'] in (3,4):
                        HDR['Excitation'] = float(np.fromfile(f, dtype=np.double, count=1)[0])
                        HDR['TriggerAdjSamples'] = int(np.fromfile(f, dtype=np.int32, count=1)[0])
                        HDR['mv_Zero'] = int(np.fromfile(f, dtype=np.int32, count=1)[0])
                        if HDR['version'] == 4:
                            HDR['WindowAverage'] = int(np.fromfile(f, dtype=np.int32, count=1)[0])
                        HDR['IniOffset'] = int(np.fromfile(f, dtype=np.int32, count=1)[0])
                    HDR['ISO_Code'] = int(str(int(np.fromfile(f, dtype=np.uint8, count=1)[0])))
                    HDR['CRC32'] = int(np.fromfile(f, dtype=np.uint32, count=1)[0])
                else:
                    raise ValueError(f'Binary format version {HDR["version"]} not recognized')
                
                # Read data
                f.seek(HDR['OffsetToData'])
                acc = np.fromfile(f, dtype=np.int16, count=int(HDR['nSamples'])).astype(np.float64)

                # Create time vector (only once)
                if data is None:  # Check if data is not initialized
                    dt = 1 / HDR['SampleRate']
                    trigger_sample_id = HDR['TriggerSampleNumber'] - start_record_sample_number
                    t = np.arange(-trigger_sample_id / HDR['SampleRate'], (HDR['nSamples'] - trigger_sample_id) / HDR['SampleRate'], dt)
                    data = np.zeros((len(t), len(dts_info) + 1)) # Pre-allocate data array with correct dimensions
                    data[:, 0] = t  # Time in the first column

                # Scale and add channel data
                if dts_info[i]['ProportionalToExcitation'].lower() == 'true' and dts_info[i]['Bridge'].lower() == 'fullbridge':
                    data[:len(acc), i + 1] = (acc - HDR['datazero_cnts']) * HDR['mv_per_cnt'] * (1 / HDR['engU_per_mV']) * (1 / float(dts_info[i]['MeasuredExcitationVoltage']))
                else:
                    data[:len(acc), i + 1] = (acc - HDR['datazero_cnts']) * HDR['mv_per_cnt'] * (1 / HDR['engU_per_mV'])

                # Append header information
                hdr.append([chan_info[i], HDR['EngUnit'], float(dts_info[i]['MeasuredExcitationVoltage'])]) # Use dts_info_entry for channel-specific info

        return data, hdr

    def import_tlf(self, filepath):
        """
        Imports binary data from a DTS G5 or DTS TDAS Pro system.

        Args:
            pn (str): Path to the directory containing the TLF and binary files.

        Returns:
            tuple: A tuple containing the data matrix and header information.
        """

        # Get directory path
        pn = os.path.dirname(filepath) + os.sep
        
        # Find TLF file
        tlf_files = [f for f in os.listdir(pn) if f.lower().endswith('.tlf')]
        if not tlf_files:
            raise FileNotFoundError("No TLF file found in the directory.")
        tlf_file = os.path.join(pn, tlf_files[0])

        # Extract channel information from TLF file
        chan_name = []
        chan_unit = []
        with open(tlf_file, 'r') as f:
            for line in f:
                if 'start sensor channel information' in line.lower():
                    # Skip next two lines
                    f.readline()
                    f.readline()
                    break
            for line in f:
                if 'posttest data' in line.lower():
                    break
                parts = line.split(',')
                chan_name.append(parts[4].strip())
                chan_unit.append(parts[15].strip())

        # Find binary files
        bin_files = [f for f in os.listdir(pn) if f.lower().endswith('.bin')]

        # Import data from binary files
        data = None
        hdr = [['Time']]  
        for i, filename in enumerate(bin_files):
            with open(os.path.join(pn, filename), 'rb') as f:
                HDR={}
                # Read header information
                HDR['acq'] = float(np.fromfile(f, dtype=np.double, count=1)[0])
                HDR['npret0'] = int(np.fromfile(f, dtype=np.int32, count=1)[0])
                HDR['npostt0'] = int(np.fromfile(f, dtype=np.int32, count=1)[0])
                HDR['prezero_cnts'] = int(np.fromfile(f, dtype=np.int32, count=1)[0])
                HDR['precal_cnts'] = int(np.fromfile(f, dtype=np.int32, count=1)[0])
                HDR['snr_dB'] = float(np.fromfile(f, dtype=np.double, count=1)[0])
                HDR['postzero_cnts'] = int(np.fromfile(f, dtype=np.int32, count=1)[0])
                HDR['postcal_cnts'] = int(np.fromfile(f, dtype=np.int32, count=1)[0])
                HDR['datazero_cnts'] = int(np.fromfile(f, dtype=np.int32, count=1)[0])
                HDR['mv_per_cnt'] = float(np.fromfile(f, dtype=np.double, count=1)[0])
                HDR['engU_per_cnt'] = float(np.fromfile(f, dtype=np.double, count=1)[0])

                # Read data
                acc = np.fromfile(f, dtype=np.int16, count=HDR['npret0'] + HDR['npostt0']).astype(np.float64)

                # Create time vector (only once)
                if data is None:  # Check if data is not initialized
                    # Create time vector
                    dt = 1 / HDR['acq']
                    t = np.arange(-HDR['npret0'] / HDR['acq'], HDR['npostt0'] / HDR['acq'], dt)
                    data = np.zeros((len(t), len(bin_files) + 1)) # Pre-allocate data array with correct dimensions
                    data[:, 0] = t  # Time in the first column

                # Scale and shift data
                data[:len(acc), i + 1] = (acc - HDR['datazero_cnts']) * HDR['engU_per_cnt']

                # Append header information
                hdr.append([chan_name[i], chan_unit[i]])

        return data, hdr

    def import_syn(self, filename):
        """Import data from Synergy (.set and .syn) files.
        
        Args:
            filename (str): Path to the .set file
            
        Returns:
            tuple: (data, hdr) where data is a numpy array and hdr is a list of headers
        """
        
        # Get directory path
        syn_dir = os.path.dirname(filename) + os.sep
        
        # Read the .set file to get channel names
        counter = 1
        idx = 0
        ch_skip = []
        ch_names = []
        
        with open(filename, 'r') as fid:
            for line in fid:
                if 'Name' in line:
                    pattern = r'<Name>(.*?)</Name>'
                    match = re.search(pattern, line)
                    if match: 
                        name = match.group(1).strip()
                    next_line = next(fid)
                    if 'true' in next_line:
                        idx += 1
                        if len(name) < 3:
                            ch_skip.append(idx)
                            continue
                        else:
                            ch_names.append(name)
                            counter += 1


        # Get list of .SYN files
        syn_files = [f for f in os.listdir(syn_dir) if f.startswith('C') and f.endswith('.SYN')]
        
        # Remove skipped channels
        for skip_idx in sorted(ch_skip, reverse=True):
            if skip_idx <= len(syn_files):
                syn_files.pop(skip_idx - 1)

        num_files = counter - 1

        # Create filenames with channel names
        fnames = []
        for i in range(num_files):
            fn = syn_files[i]
            fnames.append(f"{fn} : {ch_names[i]}")

        # Initialize data and header arrays
        data = None
        hdr = [['Time']]  # First header is always Time

        # Process each channel
        for bbb in range(num_files):
            fn = fnames[bbb].split(' : ')[0]  # Get just the filename part
            syn_path = os.path.join(syn_dir, fn)
            
            # Check file size
            if os.path.getsize(syn_path) < 600:
                messagebox.showwarning("Warning", f"Channel: {fn} appears to be empty and will be skipped")
                if num_files == 1:
                    return np.array([]), ['empty']
                continue

            # Read binary data
            with open(syn_path, 'rb') as fid:
                # Read header information
                HDR = {}
                
                HDR['STByte'] = fid.read(1).decode('ascii')
                HDR['ver'] = int.from_bytes(fid.read(1), byteorder='little')
                HDR['HDR_Length'] = int.from_bytes(fid.read(4), byteorder='little')
                HDR['overrange'] = int.from_bytes(fid.read(1), byteorder='little')
                HDR['XaxisNotTime'] = int.from_bytes(fid.read(1), byteorder='little')
                
                fid.seek(13, os.SEEK_SET)
                
                HDR['Year'] = int.from_bytes(fid.read(2), byteorder='little')
                HDR['Month'] = int.from_bytes(fid.read(1), byteorder='little')
                HDR['Day'] = int.from_bytes(fid.read(1), byteorder='little')
                HDR['Hours'] = int.from_bytes(fid.read(1), byteorder='little')
                HDR['Minutes'] = int.from_bytes(fid.read(1), byteorder='little')
                HDR['Seconds'] = int.from_bytes(fid.read(1), byteorder='little')
                HDR['DecSeconds'] = struct.unpack('d', fid.read(8))[0]
                HDR['dt'] = struct.unpack('d', fid.read(8))[0]  # Sample Interval
                HDR['TriggerPnt'] = int.from_bytes(fid.read(8), byteorder='little')
                HDR['Npts'] = int.from_bytes(fid.read(8), byteorder='little')
                
                fid.seek(56, os.SEEK_SET)
                
                HDR['NSequentialSeg'] = int.from_bytes(fid.read(4), byteorder='little')
                HDR['IntFlag'] = int.from_bytes(fid.read(1), byteorder='little')
                HDR['DataWordSize'] = int.from_bytes(fid.read(1), byteorder='little')
                HDR['ManualScale'] = int.from_bytes(fid.read(1), byteorder='little')
                HDR['Factor'] = struct.unpack('d', fid.read(8))[0]
                HDR['Offset'] = struct.unpack('d', fid.read(8))[0]
                HDR['DspMax'] = struct.unpack('d', fid.read(8))[0]
                HDR['DspMin'] = struct.unpack('d', fid.read(8))[0]
                HDR['EngUnitsEnumeration'] = int.from_bytes(fid.read(1), byteorder='little')
                HDR['FFTWindow'] = int.from_bytes(fid.read(1), byteorder='little')
                HDR['TimeSource'] = int.from_bytes(fid.read(1), byteorder='little')
                HDR['X_Offset'] = struct.unpack('d', fid.read(8))[0]

                # Read channel name
                try:
                    ch_name_bytes = fid.read(64)
                    ch_name = ''.join(chr(b) for b in ch_name_bytes[::2] if b != 0)
                    HDR['CH_Name'] = ch_name.strip()
                except Exception as e:
                    print(f"Error at: {bbb}")
                    continue

                # Read units
                y_units_bytes = fid.read(32)
                y_units = ''.join(chr(b) for b in y_units_bytes[::2] if b != 0)
                HDR['YEngUnits'] = y_units.strip()

                x_units_bytes = fid.read(32)
                x_units = ''.join(chr(b) for b in x_units_bytes[::2] if b != 0)
                HDR['XEngUnits'] = x_units.strip()

                fid.seek(576, os.SEEK_SET)

                # Read comments
                comments_bytes = fid.read(200)
                comments = ''.join(chr(b) for b in comments_bytes[::2] if b != 0)
                HDR['Comments'] = comments.strip()

                HDR['Runt'] = int.from_bytes(fid.read(1), byteorder='little')

                fid.seek(780, os.SEEK_SET)

                if not HDR['Runt']:
                    HDR['Seg0Time'] = int.from_bytes(fid.read(4), byteorder='little')
                    fid.seek(788, os.SEEK_SET)
                    dat = np.fromfile(fid, dtype=np.int16)
                else:
                    HDR['FstValidPt'] = int.from_bytes(fid.read(4), byteorder='little')
                    HDR['NValidPts'] = int.from_bytes(fid.read(4), byteorder='little')
                    HDR['FstValidSample'] = int.from_bytes(fid.read(4), byteorder='little')
                    HDR['NValidSamples'] = int.from_bytes(fid.read(4), byteorder='little')
                    HDR['Seg0Time'] = int.from_bytes(fid.read(8), byteorder='little')
                    
                    fid.seek(796, os.SEEK_SET)
                    dat = np.fromfile(fid, dtype=np.int16)

                # Construct time column
                t = np.arange(-HDR['TriggerPnt'] * HDR['dt'], 
                            (HDR['Npts'] - HDR['TriggerPnt']) * HDR['dt'], 
                            HDR['dt'])

                # Initialize data array if first channel
                if data is None:
                    data = np.zeros((len(dat), num_files + 1))
                    data[:, 0] = t[:len(dat)]  # Time column

                # Add channel data
                data[:len(dat), bbb+1] = (dat * HDR['Factor']) + HDR['Offset']

                # Add header information
                hdr.append([HDR['CH_Name'], HDR['YEngUnits']])

        return data, hdr


class SnaptoCursor:
    def __init__(self, ax, x, y_data, headers, channel_indices, fft_button_text):
        self.ax = ax
        self.x = x
        self.y_data = y_data  # List of y-data for each channel
        self.headers = headers
        self.channel_indices = channel_indices
        self.lx = ax.axhline(color='k', linewidth=0.5)  # Horizontal line
        self.ly = ax.axvline(color='k', linewidth=0.5)  # Vertical line
        self.txt = ax.text(0.05, 0.8, '', transform=ax.transAxes)  # Annotation text
        self.fft_text = fft_button_text

        # Create horizontal lines for each channel
        self.hlines = []
        for i in range(len(self.channel_indices)):
            hline = ax.axhline(color='k', linestyle='-', linewidth=0.5, visible=False)
            self.hlines.append(hline)

    def on_mouse_move(self, event):
        if not event.inaxes:
            self.lx.set_visible(False)
            self.ly.set_visible(False)
            self.txt.set_text('') 
            for hline in self.hlines:
                hline.set_visible(False)
            return

        x, y = event.xdata, event.ydata
        idx = np.searchsorted(self.x, [x])[0] 

        # Find closest data point for each channel
        closest_points = [self.find_closest_point(idx, y_channel) 
                         for y_channel in self.y_data]

        # Display cursor and annotation
        self.lx.set_data(([0, 1], [closest_points[0][1], closest_points[0][1]]))  # Update horizontal line
        self.ly.set_data(([closest_points[0][0], closest_points[0][0]], [0, 1]))  # Update vertical line
        self.lx.set_visible(True)
        self.ly.set_visible(True)

        if self.fft_text == 'Exit FFT':
            annotation_text = f"Freq: {x:.6f}\n"
        else:   
            annotation_text = f"Time: {x:.6f}\n"
        for i, (xi, yi) in enumerate(closest_points):
            channel_name = self.headers[self.channel_indices[i]][0]
            annotation_text += f"{channel_name}: {yi:.3f}\n"
            # Update horizontal line for each channel
            self.hlines[i].set_ydata([yi, yi]) 
            self.hlines[i].set_visible(True)

        self.txt.set_text(annotation_text)
        self.ax.figure.canvas.draw_idle()

    def find_closest_point(self, idx, y_data):
        # Handle edge cases where idx might be out of bounds
        idx = max(0, min(idx, len(self.x) - 1))
        return self.x[idx], y_data[idx]

    def disconnect(self):
        # Remove the lines and text completely
        self.lx.remove()
        self.ly.remove()
        self.txt.remove()
        for hline in self.hlines:
            hline.remove()
        # Delete references to these objects
        del self.lx
        del self.ly
        del self.txt
        del self.hlines

# Create and run GUI
root = tk.Tk()
app = DASQuickLookApp(root)
root.state('zoomed')  # Maximize the window
root.minsize(width=1300, height=700)
root.mainloop()