"""ecg_data.py

"""

from __future__ import print_function

import os
import sys
import math
import random
import matplotlib.pyplot as plt
import numpy as np

import wfdb


RECORD_LENGTH = 650000  # each ECG record contains 650000 samples total


# ------------------------------- ECG Records -------------------------------- #


# Return a Record object given the record number and sample period
def read_record(record_num, sampfrom=0, sampto=RECORD_LENGTH, channels=[0,1]):
	return wfdb.rdsamp('data/' + str(record_num), sampfrom=sampfrom, sampto=sampto, channels=channels)


# ----------------------------- ECG Annotations ------------------------------ #


# Return Annotation object containing only the desired beat types
def read_annotation(annotation_num, sampfrom=0, sampto=RECORD_LENGTH, beat_types=['N', 'V']):

	# Read in original file containing all annotations
	annotation = wfdb.rdann('data/' + str(annotation_num), 'atr')

	# Annotation sample locations and corresponding symbols (e.g. 'V')
	samples = annotation.sample
	symbols = annotation.symbol
	
	# Determine the sample locations and symbols of desired beats
	selected_samples = [samples[i] for i in range(len(samples)) if symbols[i] in beat_types]
	selected_symbols = [symbols[i] for i in range(len(symbols)) if symbols[i] in beat_types]

	# Write annotation file and read back in to create Annotation object
	wfdb.wrann(str(annotation_num) + '_select', 'atr', np.asarray(selected_samples), selected_symbols, fs=360)
	ecg_annotation = wfdb.rdann(str(annotation_num) + '_select', 'atr', sampfrom=sampfrom, sampto=sampto, shiftsamps=True)
	os.remove(str(annotation_num) + '_select.atr')

	return ecg_annotation


# Return Annotation object using custom data
def create_annotation(samples, symbols, sampfrom=0, sampto=RECORD_LENGTH, beat_types=['V']):

	# Determine the sample locations and symbols of desired beats
	selected_samples = [samples[i] for i in range(len(samples)) if symbols[i] in beat_types]
	selected_symbols = [symbols[i] for i in range(len(symbols)) if symbols[i] in beat_types]

	# Write annotation file and read back in to create Annotation object
	wfdb.wrann('temp_delete_me', 'atr', np.asarray(selected_samples), np.asarray(selected_symbols), fs=360)
	ecg_annotation = wfdb.rdann('temp_delete_me', 'atr', sampfrom=sampfrom, sampto=sampto, shiftsamps=True)
	os.remove('temp_delete_me.atr')

	return ecg_annotation


# -------------------------- Load ECG Record Data ---------------------------- #

# Return consecutive, sliding 2D windows for use in prediction as model input
def load_windows_2d(record_number, sampfrom=0, sampto=1080, window_radius=24):

	# Ensure that the record contains two signals (to be processed together)
	record = read_record(record_number, sampfrom, sampto)
	assert(record.nsig == 2)

	# Format the data so that it is "image-like"
	p_signals = np.array(record.p_signals.tolist())
	p_signals = np.vstack((p_signals[:,0], p_signals[:,1]))

	# Create prediction samples and corresponding windows
	samples = [i for i in range(sampfrom, sampto) if i - window_radius >= sampfrom and i + window_radius < sampto]
	windows = [p_signals[:,i - sampfrom - window_radius:i - sampfrom + window_radius + 1] for i in samples]

	return (samples, np.array(windows))


# Return train, validation, and test datasets with one-hot encoded labels
def load_datasets_windows_2d(record_number, sampfrom=0, sampto=10800, beat_types=['V'], window_radius=24):

	# Create record and annotation objects
	record = read_record(record_number, sampfrom, sampto)
	annotation = read_annotation(record_number, sampfrom, sampto, beat_types=beat_types)
	
	# Ensure that the record contains two signals (to be processed together)
	assert(record.nsig == 2)

	# Format the data so that it is "image-like"
	p_signals = np.array(record.p_signals.tolist())
	p_signals = np.vstack((p_signals[:,0], p_signals[:,1]))

	# Extract beat locations and symbols from annotation
	beat_locations = annotation.sample
	beat_symbols = annotation.symbol

	LEFT_VICINITY = 20
	RIGHT_VICINITY = 120

	# Derive PVC locations and find samples that are near a PVC beat location
	pvc_beat_locations = [beat_locations[i] for i in range(len(beat_locations)) if beat_symbols[i] == 'V']
	pvc_vicinities = [range(i - LEFT_VICINITY, i + RIGHT_VICINITY + 1) for i in pvc_beat_locations if i - LEFT_VICINITY >= sampfrom and i + RIGHT_VICINITY < sampto]
	samples_pvc = [sample for pvc_vicinity in pvc_vicinities for sample in pvc_vicinity]
	
	# Remove centers that would create incompletely captured windows due to sample range
	DOWNSAMPLE_NORMAL = 3 # downsample "normal" windows to balance out the N:V ratio
	samples = [i for i in range(sampto) if i - window_radius >= sampfrom and i + window_radius < sampto and (i in samples_pvc or i % DOWNSAMPLE_NORMAL == 0)]

	# Create examples (windows centered at samples) and labels
	num_windows = len(samples)
	x_all = [p_signals[:,i - window_radius:i + window_radius + 1] for i in samples]
	y_all = ['V' if i in samples_pvc else '-' for i in samples]

	# Randomly divide data into train, validation, and test sets
	zipped = list(zip(samples, x_all, y_all))
	random.shuffle(zipped)
	beat_locations_shuffled, x_shuffled, y_shuffled = zip(*zipped)

	divider1 = int(round(num_windows*0.70))
	divider2 = int(round(num_windows*0.90))

	beats_train, x_train, y_train = beat_locations_shuffled[:divider1], x_shuffled[:divider1], y_shuffled[:divider1]
	beats_val, x_val, y_val = beat_locations_shuffled[divider1:divider2], x_shuffled[divider1:divider2], y_shuffled[divider1:divider2]
	beats_test, x_test, y_test = beat_locations_shuffled[divider2:], x_shuffled[divider2:], y_shuffled[divider2:]

	# Sort the train, validation, and test sets
	x_train = np.array([x for _,x in sorted(zip(beats_train,x_train))])
	y_train = np.array([y for _,y in sorted(zip(beats_train,y_train))])

	x_val = np.array([x for _,x in sorted(zip(beats_val,x_val))])
	y_val = np.array([y for _,y in sorted(zip(beats_val,y_val))])

	x_test = np.array([x for _,x in sorted(zip(beats_test,x_test))])
	y_test = np.array([y for _,y in sorted(zip(beats_test,y_test))])

	# Print proportion of each beat type to total beats in each dataset
	categories = ['-'] + beat_types
	print('Record ' + str(record_number) + ':')
	for beat_type in categories:
	# for beat_type in beat_types:
		print('[' + beat_type + ' beats] Train:Validation:Test = ' + str(y_train.tolist().count(beat_type)) + ':'
										+ str(y_val.tolist().count(beat_type)) + ':'
										+ str(y_test.tolist().count(beat_type)))
	print('[*Total*] Train:Validation:Test = ' + str(len(y_train.tolist())) + ':'
										+ str(len(y_val.tolist())) + ':'
										+ str(len(y_test.tolist())) + '\n')

	# One-hot encode the ground truths
	# beat_types = np.array(beat_types)
	beat_types = np.array(categories)
	y_train = np.array([beat_types == y for y in y_train])
	y_val   = np.array([beat_types == y for y in y_val])
	y_test  = np.array([beat_types == y for y in y_test])

	# Return train, validation, and test data!
	return (x_train, y_train), (x_val, y_val), (x_test, y_test)


# --------------------------- Plot ECG Record Data --------------------------- #


# Plot ECG signal with desired annotation
def plot_prediction(record_num, ecg_annotation, sampfrom=0, sampto=10800, annstyle='r.', show_grid=True, figsize=(16,5)):

	ecg_record = read_record(record_num, sampfrom, sampto)
	plot_ecg_ann(ecg_record, ecg_annotation, record_num, sampfrom, sampto, annstyle, show_grid, figsize)

	return

# HELPER FUNCTION: Plot ECG signal given Record and Annotation objects (x-axis is in seconds)
def plot_ecg_ann(record, annotation, record_num, sampfrom=0, sampto=10800, annstyle='r*', show_grid=True, figsize=(16.5)):

	# Check the validity of items used to make the plot
	# Return the x axis time and sample values to plot for the record (and annotation if any)
	tann, annplot = checkplotitems(record, annotation)

	# Expand list styles
	annstyle = [annstyle]*record.nsig

	# Denote the ECG signals (array of each sample's amplitude)
	ecg_signals = record.p_signals

	# Denote the length and number of signals
	siglen, nsig = ecg_signals.shape

	# Using 'seconds' as the time unit
	# fs := sampling frequency
	# t = np.linspace(0, siglen-1, siglen)/record.fs

	# Shift time so that plotted time reflects actual, not shifted time
	t = np.linspace(sampfrom, sampfrom + siglen-1, siglen)/record.fs

	# Set the signal styles
	sigstyle = ['xkcd:aqua green', 'xkcd:aqua green']

	# Create the figure
	fig, axes = plt.subplots(nsig, 1, sharex=True, sharey=False, figsize=figsize)

	# Plot each channel
	for ch in range(nsig):

		# Plot signal channel
		ax = axes[ch]
		ax.plot(t, ecg_signals[:,ch], sigstyle[ch], zorder=3, linewidth=1.0)

		if ch == 0:
			ax.set_title('ECG Record ' + str(record_num))

		# Plot annotation if specified
		if annplot[ch] is not None:
			
			# Sample locations of annotations (negatives removed)
			checked_annplot = annplot[ch][annplot[ch] >= 0]

			# Time locations of annotations (negatives removed)
			checked_tann = tann[ch][tann[ch] >= 0] + sampfrom/float(record.fs)

			ax.plot(checked_tann, record.p_signals[checked_annplot, ch], annstyle[ch], zorder=3, markersize=5)

			# Original source code does not allow plotting sampfrom > 0
			# ax.plot(tann[ch], record.p_signals[annplot[ch], ch], annstyle[ch], zorder = 3)

		# Set minimum x-coordinate as beginning of actual time, not necessarily 0
		ax.set_xlim(xmin=sampfrom/float(record.fs))
		# ax.set_xlim(xmin=0)

		# Axis labels
		ax.set_ylabel(record.signame[ch] + '/' + record.units[ch])

		# Show standard ecg grids if specified
		if show_grid:

			auto_xlims = ax.get_xlim()
			auto_ylims= ax.get_ylim()

			major_ticks_x, minor_ticks_x, major_ticks_y, minor_ticks_y = calc_ecg_grids(
				auto_ylims[0], auto_ylims[1], record.units[ch], record.fs, auto_xlims[1], 'seconds')

			min_x, max_x = np.min(minor_ticks_x), np.max(minor_ticks_x)
			min_y, max_y = np.min(minor_ticks_y), np.max(minor_ticks_y)

			for tick in minor_ticks_x:
				ax.plot([tick, tick], [min_y,  max_y], linewidth=0.5, c='#ededed', marker='|', zorder=1)
			for tick in major_ticks_x:
				ax.plot([tick, tick], [min_y, max_y], linewidth=0.5, c='#bababa', marker='|', zorder=2)
			for tick in minor_ticks_y:
				ax.plot([min_x, max_x], [tick, tick], linewidth=0.5, c='#ededed', marker='_', zorder=1)
			for tick in major_ticks_y:
				ax.plot([min_x, max_x], [tick, tick], linewidth=0.5, c='#bababa', marker='_', zorder=2)

			# Plotting the lines changes the graph. Set the limits back
			ax.set_xlim(auto_xlims)
			ax.set_ylim(auto_ylims)

	fig.subplots_adjust(hspace=0.1)
	plt.xlabel('seconds')
	plt.show(fig)

	return

# HELPER FUNCTION: Calculate tick intervals for ecg grids
def calc_ecg_grids(minsig, maxsig, units, fs, maxt, timeunits):

	# 5mm 0.2s major grids, 0.04s minor grids
	# 0.5mV major grids, 0.125 minor grids 
	# 10 mm is equal to 1mV in voltage.

	# Get the grid interval of the x axis
	if timeunits == 'samples':
		majorx = 0.2*fs
		minorx = 0.04*fs
	elif timeunits == 'seconds':
		majorx = 0.2
		minorx = 0.04
	elif timeunits == 'minutes':
		majorx = 0.2/60
		minorx = 0.04/60
	elif timeunits == 'hours':
		majorx = 0.2/3600
		minorx = 0.04/3600

    # Get the grid interval of the y axis
	if units.lower()=='uv':
		majory = 500
		minory = 125
	elif units.lower()=='mv':
		majory = 0.5
		minory = 0.125
	elif units.lower()=='v':
		majory = 0.0005
		minory = 0.000125
	else:
		raise ValueError('Signal units must be uV, mV, or V to plot the ECG grid.')


	major_ticks_x = np.arange(0, upround(maxt, majorx)+0.0001, majorx)
	minor_ticks_x = np.arange(0, upround(maxt, majorx)+0.0001, minorx)

	major_ticks_y = np.arange(downround(minsig, majory), upround(maxsig, majory)+0.0001, majory)
	minor_ticks_y = np.arange(downround(minsig, majory), upround(maxsig, majory)+0.0001, minory)

	return (major_ticks_x, minor_ticks_x, major_ticks_y, minor_ticks_y)

# HELPER FUNCTION: Round down to nearest <base>
def downround(x, base):
	return base * math.floor(float(x)/base)

# HELPER FUNCTION: Round up to nearest <base>
def upround(x, base):
	return base * math.ceil(float(x)/base)

# HELPER FUNCTION: Check the validity of items used to make the plot
# Return the x-axis values (time and sample, respectively) to plot for the record (and annotation, if any)
def checkplotitems(record, annotation):
    
	siglen, nsig = record.p_signals.shape

    # Annotations if any
	if annotation is not None:

		''' This only annotates the first signal (lead)

        # The output list of numpy arrays (or Nones) to plot
		annplot = [None]*record.nsig

        # Move single channel annotations to channel 0
		annplot[0] = annotation.sample
		'''

		# Make a duplicate of the sample locations of annotations for each channel
		# All leads will therefore be annotated
		annplot = [annotation.sample] * record.nsig

        # The annotation locations to plot
		tann = [None]*record.nsig

		for ch in range(record.nsig):
			if annplot[ch] is None:
				continue
			tann[ch] = annplot[ch]/float(record.fs)

	else:
		
		tann = None
		annplot = [None]*record.nsig

	return (tann, annplot)
