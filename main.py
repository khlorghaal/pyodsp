import sounddevice as sd
from numpy import *
assert array# Make sure NumPy is loaded before it is used in the callback

import vis as vis


"""


def callback(indata, frames, time, status):
	assert(!status);
	assert(indata);

	magnitude = np.abs(np.fft.rfft(indata[:, 0], n=fftsize))
	magnitude *= args.gain / fftsize
	line = (gradient[int(np.clip(x, 0, 1) * (len(gradient) - 1))]
			for x in magnitude[low_bin:low_bin + args.columns])
	print(*line, sep='', end='\x1b[0m\n')
 
"""

"""
 if status:
			print(status, file=sys.stderr)
		global start_idx
		t = (start_idx + np.arange(frames)) / samplerate
		t = t.reshape(-1, 1)
		outdata[:] = args.amplitude * np.sin(2 * np.pi * args.frequency * t)
		start_idx += frames
"""
device_out= 5
device_in= 2
frame= 0
samplerate = sd.query_devices(device_out, 'output')['default_samplerate']

low,high= (20,2000)
columns= vis.TEX_WMAX
delta_f = (high - low) / (columns - 1)
fftsize = math.ceil(samplerate / delta_f)
low_bin = math.floor(low / delta_f)

def audio_callback(indata, outdata, frames, time, status):
	if status:
		print(status)

	

	global frame
	t= (frame+arange(frames))/samplerate
	t= t.reshape(-1, 1)
	#a= indata
	
	a= sin(t*6.28*800.)*.1
	#a= sign(a)*abs(a*a*a)

	frame+= frames
	outdata[:] = a
	vis.feed_data(a)

try:
	with sd.Stream(device=(None,None),#(None,None)
			   #samplerate=None, blocksize=None,
			   #dtype=None, latency=None,
			   channels=1,#TODO stereo?
			   latency='high',#'low'
			   callback=audio_callback):
		while vis.update(): None;
except Exception as e: raise e;


