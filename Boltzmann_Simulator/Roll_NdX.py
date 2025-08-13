import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def calculate_dice_distribution(n_dice, n_sides=6):
    """
    Calculate the number of ways to get each possible sum with n_dice dice.
    Uses dynamic programming with better numerical handling for large N.
    
    Returns:
        sums: array of possible sum values
        counts: array of number of ways to achieve each sum
        probabilities: array of probabilities for each sum
    """
    # Possible sums range from n_dice to n_dice * n_sides
    min_sum = n_dice
    max_sum = n_dice * n_sides
    
    # Initialize DP table: dp[i] = number of ways to get sum i
    dp = [0] * (max_sum + 1)
    dp[0] = 1  # Base case: one way to get sum 0 with 0 dice
    
    # For each die
    for die in range(n_dice):
        new_dp = [0] * (max_sum + 1)
        # For each possible sum so far
        for current_sum in range(len(dp)):
            if dp[current_sum] > 0:
                # Add each possible face value (1 to n_sides)
                for face in range(1, n_sides + 1):
                    if current_sum + face <= max_sum:
                        new_dp[current_sum + face] += dp[current_sum]
        dp = new_dp
    
    # Extract results for valid sums only
    sums = np.array(range(min_sum, max_sum + 1))
    counts = np.array([dp[s] for s in sums], dtype=object)  # Use object dtype for big integers
    
    # Calculate total outcomes (use Python's arbitrary precision)
    total_outcomes = n_sides ** n_dice
    
    # Convert to probabilities using floating point
    probabilities = np.array([float(count) / float(total_outcomes) for count in counts])
    
    return sums, counts, probabilities

class DiceDistributionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Dice Roll Distribution Visualizer")
        self.root.geometry("1000x700")
        
        # Variables
        self.n_dice = tk.IntVar(value=2)
        self.n_sides = tk.IntVar(value=6)
        self.y_axis_mode = tk.StringVar(value="probability")
        self.x_axis_mode = tk.StringVar(value="absolute")
        
        self.setup_gui()
        self.update_plot()
    
    def setup_gui(self):
        # Control panel frame
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Number of dice control
        ttk.Label(control_frame, text="Number of Dice (N):").pack(side=tk.LEFT, padx=(0, 5))
        dice_scale = tk.Scale(control_frame, from_=1, to=100, orient=tk.HORIZONTAL, 
                             variable=self.n_dice, command=self.on_control_change)
        dice_scale.pack(side=tk.LEFT, padx=(0, 20))
        
        # Number of sides control
        ttk.Label(control_frame, text="Sides per Die:").pack(side=tk.LEFT, padx=(0, 5))
        sides_scale = tk.Scale(control_frame, from_=2, to=20, orient=tk.HORIZONTAL, 
                              variable=self.n_sides, command=self.on_control_change)
        sides_scale.pack(side=tk.LEFT, padx=(0, 20))
        
        # Y-axis mode
        ttk.Label(control_frame, text="Y-axis:").pack(side=tk.LEFT, padx=(0, 5))
        y_frame = ttk.Frame(control_frame)
        y_frame.pack(side=tk.LEFT, padx=(0, 20))
        ttk.Radiobutton(y_frame, text="Probability", variable=self.y_axis_mode, 
                       value="probability", command=self.update_plot).pack(anchor=tk.W)
        ttk.Radiobutton(y_frame, text="Microstates", variable=self.y_axis_mode, 
                       value="counts", command=self.update_plot).pack(anchor=tk.W)
        
        # X-axis mode
        ttk.Label(control_frame, text="X-axis:").pack(side=tk.LEFT, padx=(0, 5))
        x_frame = ttk.Frame(control_frame)
        x_frame.pack(side=tk.LEFT, padx=(0, 20))
        ttk.Radiobutton(x_frame, text="Absolute Sum", variable=self.x_axis_mode, 
                       value="absolute", command=self.update_plot).pack(anchor=tk.W)
        ttk.Radiobutton(x_frame, text="Centered", variable=self.x_axis_mode, 
                       value="centered", command=self.update_plot).pack(anchor=tk.W)
        
        # Plot frame
        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Statistics frame
        stats_frame = ttk.Frame(self.root, padding="10")
        stats_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.stats_label = ttk.Label(stats_frame, text="", font=("Courier", 10))
        self.stats_label.pack()
    
    def on_control_change(self, value=None):
        """Called when any control changes"""
        self.update_plot()
    
    def update_plot(self):
        """Update the plot with current settings"""
        n_dice = self.n_dice.get()
        n_sides = self.n_sides.get()
        y_axis = self.y_axis_mode.get()
        x_axis = self.x_axis_mode.get()
        
        # Calculate distribution
        sums, counts, probabilities = calculate_dice_distribution(n_dice, n_sides)
        
        # Choose y-axis data
        if y_axis == 'probability':
            y_data = probabilities
            y_label = 'Probability'
        else:
            # For large numbers, convert to float for plotting
            y_data = np.array([float(count) for count in counts])
            y_label = 'Number of Microstates'
        
        # Choose x-axis data
        if x_axis == 'absolute':
            x_data = sums
            x_label = 'Sum'
        else:  # centered
            mean_sum = ((n_sides + 1) / 2) * n_dice  # General formula for mean
            x_data = sums - mean_sum
            x_label = 'Deviation from Mean'
        
        # Clear and plot
        self.ax.clear()
        self.ax.bar(x_data, y_data, alpha=0.7, edgecolor='black', linewidth=0.5)
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        self.ax.set_title(f'Distribution of {n_dice}d{n_sides} Rolls')
        self.ax.grid(True, alpha=0.3)
        
        # Update canvas
        self.canvas.draw()
        
        # Update statistics
        mean_sum = ((n_sides + 1) / 2) * n_dice  # General formula for mean
        most_likely_sum = sums[np.argmax(probabilities)]
        max_probability = np.max(probabilities)
        total_outcomes = n_sides ** n_dice
        
        # Format large numbers nicely
        if total_outcomes > 1e12:
            total_str = f"{total_outcomes:.2e}"
        else:
            total_str = f"{total_outcomes:,}"
        
        stats_text = (f"Statistics for {n_dice}d{n_sides}: "
                     f"Mean = {mean_sum:.1f}, "
                     f"Most likely sum = {most_likely_sum}, "
                     f"Max probability = {max_probability:.6f}, "
                     f"Total outcomes = {total_str}")
        self.stats_label.config(text=stats_text)

# Run the GUI application
if __name__ == "__main__":
    root = tk.Tk()
    app = DiceDistributionGUI(root)
    root.mainloop()