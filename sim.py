import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, Slider

# SFCW Radar Simulation

# Radar parameters
startFrequency = 500e6    # Starting frequency in Hz
stopFrequency = 2.5e9     # Stop frequency in Hz
frequencyStep = 10e6      # Frequency step in Hz
staticTargetDistance = 4  # Target distance from the radar in meters
dynamicTargetDistance = 5
sweeps = 200              # Number of frequency sweeps

EC_on = False

start_s = 80
end_sweep = 120

# Constants
c = 3e8  # Speed of light in m/s

# Calculate the frequency sweep range
frequencyRange = np.arange(startFrequency, stopFrequency + frequencyStep, frequencyStep)
numSteps = len(frequencyRange)

# Initialize arrays for storing data
timeData = np.zeros((numSteps, sweeps), dtype=complex)
rangeProfile = np.zeros((numSteps, sweeps))

# Simulate SFCW radar operation
for i in range(sweeps):
    # Calculate the time delay for the target
    timeDelay = 2 * staticTargetDistance / c
    
    if i > start_s and i < end_sweep:
        # Calculate the time delay for the dynamic target using a parabolic pattern
        middle_sweep = (start_s + end_sweep) / 2
        parabolicDelay = dynamicTargetDistance * (1 + ((i - middle_sweep) / (end_sweep - start_s))**2)
        timeDelay1 = 2 * parabolicDelay / c
    else:
        timeDelay1 = 0
    
    # Generate frequency domain data
    frequencyData = (not EC_on) * np.exp(1j * 2 * np.pi * frequencyRange * timeDelay) + 0.3 * np.exp(1j * 2 * np.pi * frequencyRange * timeDelay1)
    
    # Perform IFFT to obtain time domain data
    timeData[:, i] = np.fft.ifft(frequencyData) * np.hanning(numSteps)
    
    # Update range profile with the peak values
    rangeProfile[:, i] = np.flipud(np.abs(timeData[:, i]))

# Plot the range profile
timeAxis = np.arange(0, numSteps) * (1 / (frequencyStep * sweeps))
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 6))

# Subplot 1: Main plot
img = ax1.imshow(rangeProfile, aspect='auto', extent=[1, sweeps, timeAxis[-1], timeAxis[0]])
ax1.set_xlabel('Crossrange [cm]')
ax1.set_ylabel('Time [s]')
ax1.set_title('2D SFCW Radar Range Profile')
plt.colorbar(img, ax=ax1)

# Subplot 2: Second plot with data of the selected sweep
sweep_slider_ax = plt.axes([0.1, 0.0, 0.8, 0.03])
sweep_slider = Slider(sweep_slider_ax, 'Sweep', 0, sweeps - 1, valinit=100, valstep=1)

def update_second_plot(sweep_index):
    ax2.lines[0].set_ydata(np.flipud(np.abs(timeData[:, int(sweep_index)])))
    plt.grid()
    fig.canvas.draw()

    # Update the color of the selected sweep in the first plot
    ax1.images[0].set_clim(np.min(rangeProfile), np.max(rangeProfile))  # Set the color range
    
    # Create a mask to highlight the selected sweep
    mask = np.zeros((numSteps, sweeps))
    mask[:, int(sweep_index)] = 1
    
    # Set the color of the selected sweep to red
    ax1.images[0].set_data(np.ma.masked_array(rangeProfile, mask=mask, fill_value=np.min(rangeProfile)))
    
    fig.canvas.draw()

sweep_slider.on_changed(update_second_plot)

ax2.plot(timeAxis, np.flipud(np.abs(timeData[:, 100])))
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Amplitude')
ax2.set_title('Selected time pulse')
ax2.grid()

# Toggle EC_on variable when the switch is clicked
def toggle_EC(label):
    global EC_on
    EC_on = not EC_on
    
    # Update the range profile based on the new EC_on value
    for i in range(sweeps):
        timeDelay = 2 * staticTargetDistance / c
        
        if i > start_s and i < end_sweep:
            middle_sweep = (start_s + end_sweep) / 2
            parabolicDelay = dynamicTargetDistance * (1 + ((i - middle_sweep) / (end_sweep - start_s))**2)
            timeDelay1 = 2 * parabolicDelay / c
        else:
            timeDelay1 = 0
        
        frequencyData = (not EC_on) * np.exp(1j * 2 * np.pi * frequencyRange * timeDelay) + 0.3 * np.exp(1j * 2 * np.pi * frequencyRange * timeDelay1)
        timeData[:, i] = np.fft.ifft(frequencyData) * np.hanning(numSteps)
        rangeProfile[:, i] = np.flipud(np.abs(timeData[:, i]))
    
    # Update the main plot
    img.set_array(rangeProfile)
    
    # Update the second plot with data of the selected sweep
    sweep_index = sweep_slider.val
    ax2.lines[0].set_ydata(np.flipud(np.abs(timeData[:, int(sweep_index)])))
    plt.grid()
    fig.canvas.draw()

# Create a check button for EC_on toggle
rax = plt.axes([0.8, 0.95, 0.15, 0.05])
check_button = CheckButtons(rax, ['EC_on'], [EC_on])
check_button.on_clicked(toggle_EC)

plt.tight_layout()
plt.show()