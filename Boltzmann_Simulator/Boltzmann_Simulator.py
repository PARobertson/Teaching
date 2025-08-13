import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
import os

# Physical constants
k_B = 1.38064852e-23  # Boltzmann constant (J/K)
h = 6.62607015e-34    # Planck constant (J⋅s)
c = 2.99792458e8      # Speed of light (m/s)

def wavenumber_to_joules(wavenumber_cm):
    """Convert wavenumber (cm⁻¹) to energy (J)"""
    return h * c * wavenumber_cm * 100

class MolecularDatabase:
    def __init__(self):
        self.db_file = "molecules.json"
        self.default_molecules = {
            "HCl": {"omega_e": 2886, "B_e": 10.59, "alpha_e": 0.307, "sigma": 1, "description": "Hydrogen chloride"},
            "CO": {"omega_e": 2170, "B_e": 1.93, "alpha_e": 0.0175, "sigma": 1, "description": "Carbon monoxide"},
            "H2": {"omega_e": 4401, "B_e": 60.9, "alpha_e": 3.06, "sigma": 2, "description": "Hydrogen"},
            "N2": {"omega_e": 2359, "B_e": 2.01, "alpha_e": 0.0179, "sigma": 2, "description": "Nitrogen"},
            "O2": {"omega_e": 1580, "B_e": 1.44, "alpha_e": 0.0159, "sigma": 2, "description": "Oxygen"},
            "NO": {"omega_e": 1904, "B_e": 1.70, "alpha_e": 0.017, "sigma": 1, "description": "Nitric oxide"},
            "I2": {"omega_e": 214, "B_e": 0.037, "alpha_e": 0.00014, "sigma": 2, "description": "Iodine"},
            "Custom": {"omega_e": 1000, "B_e": 1.0, "alpha_e": 0.01, "sigma": 1, "description": "Custom parameters"}
        }
        self.load_database()
    
    def load_database(self):
        try:
            if os.path.exists(self.db_file):
                with open(self.db_file, 'r') as f:
                    self.molecules = json.load(f)
            else:
                self.molecules = self.default_molecules.copy()
                self.save_database()
        except Exception:
            self.molecules = self.default_molecules.copy()
    
    def save_database(self):
        try:
            with open(self.db_file, 'w') as f:
                json.dump(self.molecules, f, indent=2)
        except Exception:
            pass
    
    def get_molecule_names(self):
        return list(self.molecules.keys())
    
    def get_parameters(self, molecule_name):
        return self.molecules.get(molecule_name, self.molecules["Custom"])
    
    def add_molecule(self, name, omega_e, B_e, alpha_e, sigma, description=""):
        self.molecules[name] = {
            "omega_e": float(omega_e),
            "B_e": float(B_e), 
            "alpha_e": float(alpha_e),
            "sigma": int(sigma),
            "description": description
        }
        self.save_database()

def calculate_harmonic_oscillator_distribution(omega_cm, temperature, threshold=1e-9):
    hc_omega = wavenumber_to_joules(omega_cm)
    
    if temperature <= 0:
        n_levels = np.array([0])
        energies_cm = np.array([omega_cm * 0.5])
        populations = np.array([1.0])
        boltzmann_factors = np.array([1.0])
        partition_function = 1.0
        return n_levels, energies_cm, populations, boltzmann_factors, partition_function
    
    beta = 1.0 / (k_B * temperature)
    n_levels, energies_j, boltzmann_factors = [], [], []
    
    n = 0
    while True:
        energy_j = hc_omega * n  # neglecting ZPE
        boltz_factor = np.exp(-energy_j * beta)
        
        if n == 0:
            ground_state_factor = boltz_factor
            min_factor = ground_state_factor * threshold
        
        if n > 0 and boltz_factor < min_factor:
            break
        if n > 100:
            break
        
        n_levels.append(n)
        energies_j.append(energy_j)
        boltzmann_factors.append(boltz_factor)
        n += 1
    
    n_levels = np.array(n_levels)
    energies_j = np.array(energies_j)
    boltzmann_factors = np.array(boltzmann_factors)
    energies_cm = energies_j / wavenumber_to_joules(1.0)
    
    partition_function = np.sum(boltzmann_factors)
    populations = boltzmann_factors / partition_function
    
    return n_levels, energies_cm, populations, boltzmann_factors, partition_function

def calculate_rigid_rotor_distribution(B_cm, temperature, symmetry_number=1, threshold=1e-6):
    hc_B = wavenumber_to_joules(B_cm)
    
    if temperature <= 0:
        J_levels = np.array([0])
        energies_cm = np.array([0.0])
        populations = np.array([1.0])
        boltzmann_factors = np.array([1.0])
        partition_function = 1.0
        return J_levels, energies_cm, populations, boltzmann_factors, partition_function
    
    beta = 1.0 / (k_B * temperature)
    J_levels, energies_j, boltzmann_factors = [], [], []
    
    J = 0
    while True:
        energy_j = hc_B * J * (J + 1)
        degeneracy = (2 * J + 1) / symmetry_number
        boltz_factor = degeneracy * np.exp(-energy_j * beta)
        
        if J == 0:
            ground_state_factor = boltz_factor
            min_factor = ground_state_factor * threshold
        
        if J > 4 and boltz_factor < min_factor:
            break
        if J > 200:
            break
        
        J_levels.append(J)
        energies_j.append(energy_j)
        boltzmann_factors.append(boltz_factor)
        J += 1
    
    J_levels = np.array(J_levels)
    energies_j = np.array(energies_j)
    boltzmann_factors = np.array(boltzmann_factors)
    energies_cm = energies_j / wavenumber_to_joules(1.0)
    
    partition_function = np.sum(boltzmann_factors)
    populations = boltzmann_factors / partition_function
    
    return J_levels, energies_cm, populations, boltzmann_factors, partition_function

class BoltzmannDistributionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Quantum Boltzmann Distribution Visualizer")
        self.root.geometry("1200x700")
        
        # Initialize database
        self.db = MolecularDatabase()
        
        # Variables
        self.temperature = tk.DoubleVar(value=300.0)
        self.selected_molecule = tk.StringVar(value="HCl")
        self.omega_e_var = tk.StringVar()
        self.B_e_var = tk.StringVar()
        self.alpha_e_var = tk.StringVar()
        self.sigma_var = tk.StringVar()
        self.mode = tk.StringVar(value="harmonic")  # "harmonic", "rotor"
        self.y_axis_mode = tk.StringVar(value="populations")
        
        self.setup_gui()
        self.load_molecule_parameters()
        self.update_interface()
        self.update_plot()
    
    def setup_gui(self):
        # Top control frame
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Mode selection
        mode_frame = ttk.LabelFrame(control_frame, text="System Type", padding="5")
        mode_frame.pack(side=tk.LEFT, padx=(0, 10), fill=tk.Y)
        
        ttk.Radiobutton(mode_frame, text="Harmonic Oscillator", 
                       variable=self.mode, value="harmonic", command=self.on_mode_change).pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="Rigid Rotor", 
                       variable=self.mode, value="rotor", command=self.on_mode_change).pack(anchor=tk.W)
        
        # Molecule selection
        mol_frame = ttk.LabelFrame(control_frame, text="Molecule", padding="5")
        mol_frame.pack(side=tk.LEFT, padx=(0, 10), fill=tk.Y)
        
        self.molecule_combo = ttk.Combobox(mol_frame, textvariable=self.selected_molecule,
                                          values=self.db.get_molecule_names(), state="readonly", width=8)
        self.molecule_combo.pack(pady=(0, 5))
        self.molecule_combo.bind('<<ComboboxSelected>>', self.on_molecule_change)
        
        ttk.Button(mol_frame, text="Add Molecule", command=self.add_molecule, width=12).pack()
        
        # Temperature
        temp_frame = ttk.LabelFrame(control_frame, text="Temperature", padding="5")
        temp_frame.pack(side=tk.LEFT, padx=(0, 10), fill=tk.Y)
        
        temp_entry = ttk.Entry(temp_frame, textvariable=self.temperature, width=8)
        temp_entry.pack(pady=(0, 2))
        ttk.Label(temp_frame, text="Kelvin").pack()
        
        temp_scale = tk.Scale(temp_frame, from_=0, to=10000, orient=tk.HORIZONTAL,
                             variable=self.temperature, length=150)
        temp_scale.pack()
        
        # Parameters frame (will be populated dynamically)
        self.param_frame = ttk.LabelFrame(control_frame, text="Parameters", padding="5")
        self.param_frame.pack(side=tk.LEFT, padx=(0, 10), fill=tk.Y)
        
        # Options frame
        self.options_frame = ttk.LabelFrame(control_frame, text="Y-axis", padding="5")
        self.options_frame.pack(side=tk.LEFT, padx=(0, 10), fill=tk.Y)
        
        ttk.Radiobutton(self.options_frame, text="Populations", 
                       variable=self.y_axis_mode, value="populations").pack(anchor=tk.W)
        ttk.Radiobutton(self.options_frame, text="Boltzmann factors", 
                       variable=self.y_axis_mode, value="boltzmann").pack(anchor=tk.W)
        
        # Update button
        update_frame = ttk.LabelFrame(control_frame, text="Update", padding="5")
        update_frame.pack(side=tk.LEFT, padx=(0, 10), fill=tk.Y)
        
        ttk.Button(update_frame, text="UPDATE\nPLOT", command=self.update_plot, width=10).pack(expand=True)

        # Statistics
        stats_frame = ttk.Frame(self.root, padding="10")
        stats_frame.pack(side=tk.TOP, fill=tk.X)
        
        self.stats_label = ttk.Label(stats_frame, text="", font=("Courier", 10))
        self.stats_label.pack()

        # Plot area
        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        

    
    def load_molecule_parameters(self):
        mol_name = self.selected_molecule.get()
        params = self.db.get_parameters(mol_name)
        
        self.omega_e_var.set(str(params["omega_e"]))
        self.B_e_var.set(str(params["B_e"]))
        self.alpha_e_var.set(str(params["alpha_e"]))
        self.sigma_var.set(str(params["sigma"]))
    
    def on_molecule_change(self, event=None):
        self.load_molecule_parameters()
        self.update_interface()
    
    def on_mode_change(self):
        self.update_interface()
    
    def update_interface(self):
        # Clear existing widgets
        for widget in self.param_frame.winfo_children():
            widget.destroy()
        
        mode = self.mode.get()
        
        # Show current molecular parameters (editable for Custom)
        mol_name = self.selected_molecule.get()
        if mol_name == "Custom":
            # Editable entries for custom parameters
            if mode == "harmonic":
                ttk.Label(self.param_frame, text="ωₑ (cm⁻¹):").pack(anchor=tk.W)   
                omega_entry = ttk.Entry(self.param_frame, textvariable=self.omega_e_var, width=10)
                omega_entry.pack(anchor=tk.W, pady=(0, 5))

            elif mode == "rotor":
                ttk.Label(self.param_frame, text="Bₑ (cm⁻¹):").pack(anchor=tk.W)
                B_entry = ttk.Entry(self.param_frame, textvariable=self.B_e_var, width=10)
                B_entry.pack(anchor=tk.W, pady=(0, 5))
                
                ttk.Label(self.param_frame, text="σ:").pack(anchor=tk.W)
                sigma_entry = ttk.Entry(self.param_frame, textvariable=self.sigma_var, width=10)
                sigma_entry.pack(anchor=tk.W, pady=(0, 5))
        else:
            # Display-only labels for database molecules
            if mode == "harmonic":
                ttk.Label(self.param_frame, text=f"ωₑ: {self.omega_e_var.get()} cm⁻¹").pack(anchor=tk.W)
                vib_temp = wavenumber_to_joules(float(self.omega_e_var.get())) / k_B
                ttk.Label(self.param_frame, text=f"θ_vib: {vib_temp:.0f} K").pack(anchor=tk.W)   
            elif mode == "rotor":
                ttk.Label(self.param_frame, text=f"Bₑ: {self.B_e_var.get()} cm⁻¹").pack(anchor=tk.W)
                ttk.Label(self.param_frame, text=f"σ: {self.sigma_var.get()}").pack(anchor=tk.W)
                rot_temp = wavenumber_to_joules(float(self.B_e_var.get())) / k_B
                ttk.Label(self.param_frame, text=f"θ_rot: {rot_temp:.0f} K").pack(anchor=tk.W)
                
    def add_molecule(self):
        dialog = MoleculeDialog(self.root, "Add Molecule")
        if dialog.result:
            name, omega_e, B_e, alpha_e, sigma, description = dialog.result
            self.db.add_molecule(name, omega_e, B_e, alpha_e, sigma, description)
            self.molecule_combo['values'] = self.db.get_molecule_names()
            self.selected_molecule.set(name)
            self.load_molecule_parameters()
            self.update_interface()
    
    def update_plot(self):
        try:
            mode = self.mode.get()
            temperature = self.temperature.get()
            omega_e = float(self.omega_e_var.get())
            B_e = float(self.B_e_var.get())
            symmetry = int(self.sigma_var.get())
            
            self.ax.clear()
            
            if mode == "harmonic":
                levels, energies_cm, populations, boltzmann_factors, partition_function = calculate_harmonic_oscillator_distribution(
                    omega_e, temperature)
                
                y_data = populations if self.y_axis_mode.get() == "populations" else boltzmann_factors
                y_label = 'Population Probability' if self.y_axis_mode.get() == "populations" else 'Boltzmann Factor'
                
                bars = self.ax.bar(levels, y_data, alpha=0.7, edgecolor='black', linewidth=0.5)
                
                # Color bars by population intensity
                if len(populations) > 0:
                    max_pop = np.max(populations)
                    for bar, pop in zip(bars, populations):
                        intensity = pop / max_pop if max_pop > 0 else 0
                        bar.set_facecolor((1-intensity, 0.5, intensity))
                
                self.ax.set_xlabel('Vibrational Quantum Number (n)')
                self.ax.set_xticks(levels)
                self.ax.set_ylabel(y_label)
                mol_name = self.selected_molecule.get()
                self.ax.set_title(f'{mol_name} Harmonic Oscillator: T = {temperature:.0f} K')
                
                # Statistics
                avg_level = np.sum(levels * populations)
                ground_state_pop = populations[0]
                kt_ratio = (k_B * temperature) / wavenumber_to_joules(omega_e) if temperature > 0 else 0
                                
                stats_text = (f"⟨n⟩ = {avg_level:.3f}, P(0): {ground_state_pop:.3f}, "
                             f"k_BT/(hcv) = {kt_ratio:.3f}, q_vib = {partition_function:.2f}")
            
            elif mode == "rotor":
                levels, energies_cm, populations, boltzmann_factors, partition_function = calculate_rigid_rotor_distribution(
                    B_e, temperature, symmetry)
                
                y_data = populations if self.y_axis_mode.get() == "populations" else boltzmann_factors
                y_label = 'Population Probability' if self.y_axis_mode.get() == "populations" else 'Boltzmann Factor'
                
                bars = self.ax.bar(levels, y_data, alpha=0.7, edgecolor='black', linewidth=0.5)
                
                # Color bars by population intensity
                if len(populations) > 0:
                    max_pop = np.max(populations)
                    for bar, pop in zip(bars, populations):
                        intensity = pop / max_pop if max_pop > 0 else 0
                        bar.set_facecolor((intensity, 0.5, 1-intensity))
                
                self.ax.set_xlabel('Rotational Quantum Number (J)')
                self.ax.set_xticks(levels)
                self.ax.set_ylabel(y_label)
                mol_name = self.selected_molecule.get()
                self.ax.set_title(f'{mol_name} Rigid Rotor: T = {temperature:.0f} K, σ = {symmetry}')
                
                # Statistics
                avg_level = np.sum(levels * populations)
                ground_state_pop = populations[0]
                kt_ratio = (k_B * temperature) / wavenumber_to_joules(B_e) if temperature > 0 else 0
                
                stats_text = (f"⟨J⟩ = {avg_level:.3f}, Ground state: {ground_state_pop:.3f}, "
                             f"k_BT/(ℏcB) = {kt_ratio:.3f}, Z_rot = {partition_function:.2f}, Levels: {len(levels)}")
            
            # Add grid and set log scale if needed
            self.ax.grid(True, alpha=0.3)
            if self.y_axis_mode.get() == 'boltzmann' and temperature > 0:
                self.ax.set_yscale('log')
            
            self.canvas.draw()
            self.stats_label.config(text=stats_text)
            
        except Exception as e:
            print(f"Error in update_plot: {e}")

class MoleculeDialog:
    def __init__(self, parent, title):
        self.result = None
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("350x250")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self.dialog.geometry("+%d+%d" % (parent.winfo_rootx() + 50, parent.winfo_rooty() + 50))
        
        self.name_var = tk.StringVar()
        self.omega_var = tk.StringVar(value="1000")
        self.B_var = tk.StringVar(value="1.0")
        self.alpha_var = tk.StringVar(value="0.01")
        self.sigma_var = tk.StringVar(value="1")
        self.desc_var = tk.StringVar()
        
        self.create_widgets()
    
    def create_widgets(self):
        main_frame = ttk.Frame(self.dialog, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        fields = [
            ("Name:", self.name_var),
            ("ωₑ (cm⁻¹):", self.omega_var),
            ("Bₑ (cm⁻¹):", self.B_var),
            ("αₑ (cm⁻¹):", self.alpha_var),
            ("σ:", self.sigma_var),
            ("Description:", self.desc_var)
        ]
        
        for i, (label, var) in enumerate(fields):
            ttk.Label(main_frame, text=label).grid(row=i, column=0, sticky=tk.W, pady=3)
            entry = ttk.Entry(main_frame, textvariable=var, width=15)
            entry.grid(row=i, column=1, pady=3, padx=(10, 0))
        
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=len(fields), column=0, columnspan=2, pady=15)
        
        ttk.Button(btn_frame, text="OK", command=self.ok_clicked).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(btn_frame, text="Cancel", command=self.cancel_clicked).pack(side=tk.LEFT)
    
    def ok_clicked(self):
        try:
            name = self.name_var.get().strip()
            if not name:
                messagebox.showerror("Error", "Name required")
                return
            
            omega_e = float(self.omega_var.get())
            B_e = float(self.B_var.get())
            alpha_e = float(self.alpha_var.get())
            sigma = int(self.sigma_var.get())
            description = self.desc_var.get().strip()
            
            if omega_e <= 0 or B_e <= 0 or sigma <= 0:
                messagebox.showerror("Error", "ωₑ, Bₑ, and σ must be positive")
                return
            
            self.result = (name, omega_e, B_e, alpha_e, sigma, description)
            self.dialog.destroy()
            
        except ValueError:
            messagebox.showerror("Error", "Invalid values")
    
    def cancel_clicked(self):
        self.dialog.destroy()

if __name__ == "__main__":
    import sys
    if 'ipykernel' in sys.modules:
        # Running in Jupyter - need special handling for tkinter
        print("Note: For best experience, download and run locally")
        print("Tkinter GUIs work best outside of browser environments")
    else:
        # Running normally
        root = tk.Tk()
        app = BoltzmannDistributionGUI(root)
        root.mainloop()