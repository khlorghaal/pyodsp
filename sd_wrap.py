#handles audio config, buffers
#provides a bit more function than just sounddevice
#while hiding common stuff like sample rate

from com import *
import vis
import sounddevice as sd
#DEBUG= True
#if DEBUG:
sdqd= sd.query_devices()
print(sdqd)

#(out,in)
voicemeeter= False
for s in sdqd:
	if 'Voicemeeter' in s:
		voicemeeter= True
if voicemeeter:
	sd.default.device= 21 #voicemetter
	DUPLEX= True
else:
	#sd.default.device= (4,1)#normal
	#sd.default.device= (2,6)#???
    sd.default.device= (12,12)#linux
    DUPLEX= True
    #DUPLEX= False #whether duplex really fucking depends

#must be pure function, as async
from enum import Enum
class audio_op:
	instance= None
	class arity(Enum):
		AMP_OUT=1
		AMP_INOUT=2
		FFT_OUT=3
		FFT_INOUT=4
	#[sig sd_wrap.audio_op; bool ]
	def __init__(self,arity,f):#todo does meta-fft also require windowing?
		self.arity= arity
		self.f=     f

from numpy import complex64
from numpy import *
from numpy.fft import *
import scipy.fftpack as fftpack
sample_rate = sd.query_devices(sd.default.device if DUPLEX else sd.default.device[0], 'input')['default_samplerate']
#fftlen= fftpack.next_fast_len(fftsize_calc(sample_rate))
#1:1 size of samples:fft
# otherwise rescaling is required, which is pessimum
#samples= zeros(fftlen)
#fft= zeros(fftlen)
def push_samples(arr):
	global samples
	o= arr.size
	assert o<sample_count
	samples[o:-1]= samples[0:-o-1]
	samples[0:o]= arr
	return samples#*hann(sample_count)*800

def hz_idx(t):
	return (t*sample_count/sample_rate).astype(int)
frame= 0
t0= 0
last_amp= 0# prevents skipping on dropped frames
def audio_callback(indata, outdata, frames, time, status):
	#ASYNC this function is invoked from another thread
	if DEBUG:
		print('______update')
	if status:
		print('STATUS'+str(status))
		print('frames'+str(frames))
	global frame
	global data_p
	global data_pp
	global t0

	stereo= indata.shape[1]==2
	#if DEBUG:
		#print('note_array.shape '+note_array.shape)

	in_amp=   indata[:,0].view()
	in_fft= rfft(in_amp)
	out_amp= outdata[:,0].view()
	out_frq= zeros(frames//2,dtype=complex64)

	t1= t0+float(frames)/sample_rate
	if DEBUG:
		print('interval '+str(t0-t1))

	#DO NOT high latency ops such as allocation, because underflow
	#all bulk ops should be as o[:]=... to prevent realloc
	#todo theres a few copy ops which may or may not need eliminated
	#i believe double buffering wouldnt aid much

	#amplitude
	#freq= resample(freq, int(freq.size*2.))
	#freq= freq[:max(1600,fftsize)] #denoise
	#freq= fft(freq, n=fftsize)
	#a= ifft(roll(freq,200.), n=fftsize)
	##a= irfft(a[::], n=frames) ????

	#todo signal window functions

	#optimize flywheel - elim allocations


	visin= (in_amp.copy(),in_fft.copy())


	_op= audio_op.instance
	do_rfft= False #delaying allows postprocess
	def _AMP_OUT():
		out_amp[:]= _op.f( zeros(frames) )
		out_frq[:]= rfft(out_amp)
	def _AMP_INOUT():
		out_amp[:]= _op.f( in_amp )
		out_frq[:]= rfft(out_amp)
	def _FFT_OUT():
		out_frq[:]= _op.f( zeros(frames) )
		do_rfft= True
	def _FFT_INOUT():
		out_frq[:]= _op.f( in_fft )
		do_rfft= True
	{
		audio_op.arity.AMP_OUT: _AMP_OUT,
		audio_op.arity.AMP_INOUT: _AMP_INOUT,
		audio_op.arity.FFT_OUT: _FFT_OUT,
		audio_op.arity.FFT_INOUT: _FFT_INOUT,
	}[_op.arity]()


	visout= (out_amp.copy(),out_frq.copy())
		#transforms after here are not shown on graphs

	#temp window function
	#evited by using sine
	# has slight low freq emission but prevents all popping
	#o[:]*= pow( linspace(0,2,o.size)*linspace(2,0,o.size), 2.)

	#lowpass
	_lk= 10#half-frequency, as in exponential, same as half-life
	out_frq*= square(_lk/arange(_lk,_lk+out_frq.size)) # 1 -> lim 0
	out_amp[:]= irfft(out_frq*1j, n=len(out_amp))
		# *1j does cosine->sine, to taper ends and mostly elim need for windowing


	MIXMIC= False
	if MIXMIC:
		out_amp[:]+= in_amp

	if stereo:#chanel mirroring
		outdata[:,1]= outdata[:,0]

	vis.fifo.put_nowait((*visin, *visout))
	#copy is unevitable since o==outdata are managed by outer scope
	#	here was determined to be the appropriate location to copy
	#	manually buffering would still require a copy
	t0= t1
	frame+= frames



#parameters are for different threads
#dont let them fuck with eachother
def invoke(update_main, update_audio):
	audio_op.instance= update_audio
	try:
		with sd.Stream(
			#samplerate=sample_rate,
			samplerate=44000,
			#blocksize=fftsize,
			blocksize=2048,
			#dtype=None, latency=None,
			clip_off=True,
			dither_off= True,
			##never_drop_input= True,
			channels=1,#mono input and output
			#latency='high',
			callback= audio_callback):
				while update_main(): None;
	except Exception as e: 
		print("\nEXCEPT")
	    #always check if devices are configured via sd.default.device
		raise e