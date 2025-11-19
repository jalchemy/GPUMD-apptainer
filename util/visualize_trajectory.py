
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D

# This script requires ase, matplotlib, and numpy.
# You can install them using pip:
# pip install ase matplotlib numpy

try:
    from ase.io import read
except ImportError:
    print("Error: ASE (Atomic Simulation Environment) is not installed.")
    print("Please install it using: pip install ase")
    sys.exit(1)


class TrajectoryVisualizer:
    """
    A class to visualize molecular dynamics trajectories from an extended XYZ file.
    It displays atomic positions in a 3D plot and the system's energy over time,
    with interactive controls for animation.
    """

    def __init__(self, frames, xyz_file_path):
        if not frames:
            print("Error: No frames to visualize.")
            sys.exit(1)

        self.frames = frames
        self.xyz_file_path = xyz_file_path
        self.num_frames = len(frames)
        self.current_frame_idx = 0
        self.playing = False
        self.timer = None

        # Extract data from frames
        self.energies = [frame.get_potential_energy() for frame in self.frames]
        self.times = [frame.info.get('Time', i) for i, frame in enumerate(self.frames)]
        
        # Determine atom coloring
        all_symbols = sorted(list(set(frames[0].get_chemical_symbols())))
        cmap = plt.get_cmap('viridis')
        self.symbol_color_map = {symbol: cmap(i / (len(all_symbols) - 1)) for i, symbol in enumerate(all_symbols)}

        # Set up the plot
        self.fig = plt.figure(figsize=(16, 8))
        self.fig.suptitle(f"Trajectory Visualization: {self.xyz_file_path}", fontsize=16)
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1, 1])
        self.ax_3d = self.fig.add_subplot(gs[0], projection='3d')
        self.ax_energy = self.fig.add_subplot(gs[1])
        
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.2, top=0.9)

        self._setup_3d_plot()
        self._setup_energy_plot()
        self._setup_widgets()

        self.update(0)
        self._set_axis_limits()

    def _setup_3d_plot(self):
        """Initial setup of the 3D scatter plot."""
        self.ax_3d.set_title("Atomic Positions")
        self.ax_3d.set_xlabel("X (Å)")
        self.ax_3d.set_ylabel("Y (Å)")
        self.ax_3d.set_zlabel("Z (Å)")
        
        # Create scatter plot artists for each atom type
        self.scatter_plots = {}
        initial_atoms = self.frames[0]
        symbols = initial_atoms.get_chemical_symbols()
        positions = initial_atoms.get_positions()

        for symbol in self.symbol_color_map:
            indices = [i for i, s in enumerate(symbols) if s == symbol]
            if indices:
                pos = positions[indices]
                self.scatter_plots[symbol] = self.ax_3d.scatter(
                    pos[:, 0], pos[:, 1], pos[:, 2],
                    color=self.symbol_color_map[symbol],
                    label=symbol,
                    s=40,
                    alpha=0.8
                )
        self.ax_3d.legend()

    def _set_axis_limits(self):
        """Set fixed axis limits based on the entire trajectory."""
        all_pos = np.vstack([f.get_positions() for f in self.frames])
        min_coords = all_pos.min(axis=0)
        max_coords = all_pos.max(axis=0)
        center = (max_coords + min_coords) / 2
        max_range = (max_coords - min_coords).max() * 0.6
        
        self.ax_3d.set_xlim(center[0] - max_range, center[0] + max_range)
        self.ax_3d.set_ylim(center[1] - max_range, center[1] + max_range)
        self.ax_3d.set_zlim(center[2] - max_range, center[2] + max_range)

    def _setup_energy_plot(self):
        """Setup of the energy plot."""
        self.ax_energy.plot(self.times, self.energies, lw=2, label="Potential Energy")
        self.energy_marker, = self.ax_energy.plot(
            self.times[0], self.energies[0], 'o', markersize=10, color='red'
        )
        self.ax_energy.set_xlabel("Time")
        self.ax_energy.set_ylabel("Potential Energy (eV)")
        self.ax_energy.set_title("Energy Profile")
        self.ax_energy.grid(True)
        self.ax_energy.legend()

    def _setup_widgets(self):
        """Setup of the slider and buttons."""
        ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
        self.slider = Slider(
            ax=ax_slider,
            label='Frame',
            valmin=0,
            valmax=self.num_frames - 1,
            valinit=0,
            valstep=1
        )
        self.slider.on_changed(self.update)

        btn_props = {'color': 'lightgoldenrodyellow', 'hovercolor': '0.975'}
        ax_prev = plt.axes([0.30, 0.025, 0.1, 0.04])
        self.btn_prev = Button(ax_prev, '<< Prev', **btn_props)
        self.btn_prev.on_clicked(self._prev_frame)

        ax_play = plt.axes([0.41, 0.025, 0.1, 0.04])
        self.btn_play = Button(ax_play, 'Play >', **btn_props)
        self.btn_play.on_clicked(self._play)

        ax_pause = plt.axes([0.52, 0.025, 0.1, 0.04])
        self.btn_pause = Button(ax_pause, 'Pause ||', **btn_props)
        self.btn_pause.on_clicked(self._pause)

        ax_next = plt.axes([0.63, 0.025, 0.1, 0.04])
        self.btn_next = Button(ax_next, 'Next >>', **btn_props)
        self.btn_next.on_clicked(self._next_frame)

    def update(self, frame_idx):
        """Updates the plot to a specific frame."""
        frame_idx = int(frame_idx)
        self.current_frame_idx = frame_idx
        
        atoms = self.frames[frame_idx]
        positions = atoms.get_positions()
        symbols = atoms.get_chemical_symbols()

        # Update 3D scatter plot data
        for symbol, scatter_plot in self.scatter_plots.items():
            indices = [i for i, s in enumerate(symbols) if s == symbol]
            if indices:
                pos = positions[indices]
                scatter_plot._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])

        # Update energy marker
        self.energy_marker.set_data(self.times[frame_idx], self.energies[frame_idx])
        
        self.ax_3d.set_title(f"Atomic Positions (Frame {frame_idx})")

        # Update slider if change was triggered by animation
        if self.slider.val != frame_idx:
            self.slider.set_val(frame_idx)

        self.fig.canvas.draw_idle()

    def _next_frame(self, event=None):
        next_idx = (self.current_frame_idx + 1) % self.num_frames
        self.slider.set_val(next_idx)

    def _prev_frame(self, event=None):
        prev_idx = (self.current_frame_idx - 1 + self.num_frames) % self.num_frames
        self.slider.set_val(prev_idx)

    def _play(self, event=None):
        if self.playing:
            return
        self.playing = True
        self.timer = self.fig.canvas.new_timer(interval=100)  # 10 FPS
        self.timer.add_callback(self._animation_step)
        self.timer.start()

    def _pause(self, event=None):
        if self.playing:
            self.timer.stop()
            self.playing = False

    def _animation_step(self):
        """A single step of the animation, called by the timer."""
        if self.current_frame_idx == self.num_frames - 1:
            self._pause()  # Stop at the end
        else:
            self._next_frame()


def main():
    """
    Main function to load data and run the visualizer.
    """
    if len(sys.argv) > 2:
        print("Usage: python visualize_trajectory.py [<path_to_xyz_file>]")
        sys.exit(1)

    xyz_file = None
    if len(sys.argv) == 2:
        xyz_file = sys.argv[1]
    else:
        # Fallback to a default file if no argument is provided
        default_file = "example/model.xyz"
        print(f"No file path provided. Trying default: '{default_file}'")
        xyz_file = default_file

    try:
        print(f"Loading trajectory from '{xyz_file}'...")
        frames = read(xyz_file, index=':')
        print(f"Successfully loaded {len(frames)} frames.")
    except FileNotFoundError:
        print(f"Error: File not found at '{xyz_file}'")
        print("Please provide a valid path to an extended XYZ file.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        sys.exit(1)

    # Create and show the visualizer
    visualizer = TrajectoryVisualizer(frames, xyz_file)
    plt.show()


if __name__ == "__main__":
    main()
