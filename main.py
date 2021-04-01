'''
stereo wouldnt even work
	since the audio channels have a nonlinear coupling which cannot be modeled easily
	maybe later i should say fuckit and see if it gets decent results regardless
	visualiser also would need to accomodate stereo or warn its displaying mono


todo
wtf am doing with visualizer
	waveform : fft :: immediate raw, multiscale : time-accumulated
blend accumulation visualizer

piano
	timbre modulation

voicemod
	fractal overlay
		speed and frequency
'''

from com import *

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
    sd.default.device= (11,11)#linux
    #DUPLEX= True
    DUPLEX= False #whether duplex really fucking depends
    #im not exactly sure what a duplex even is

import vis

from numpy import *
from numpy.fft import *
import scipy.fftpack as fftpack

from scipy.interpolate import splrep,splev
from scipy.signal import resample,resample_poly
from scipy.signal.windows import hann, triang

import time
import pygame
import pygame.key
from pygame.locals import *

from math import  pi as PI
from math import tau as TAU



def sample(arr,mul):
	up=   mul if mul>1. else 1.
	down= 1.  if mul>1. else 1./mul
	return resample_poly(arr,up,down)
	"""
	if(mul<1.):#minifier

		return arr
	elif(mul>1.):#cubic interpolation
		x= arange(arr.size)
		s= splrep(x,arr)
		return splev(x,s,der=0)
	return arr;
	"""
def saturate(arr):
	return clip(arr,0,1)


#input
note_keys= array([
		[K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, K_9, K_0],
		[K_q, K_w, K_e, K_r, K_t, K_y, K_u, K_i, K_o, K_p],
		[K_a, K_s, K_d, K_f, K_g, K_h, K_j, K_k, K_l, K_SEMICOLON],
		[K_z, K_x, K_c, K_v, K_b, K_n, K_m, K_COMMA,  K_PERIOD, K_SLASH]
	])
note_state= zeros(note_keys.shape)

keymap={}#keycode -> 2d position
for i,row in enumerate(note_keys):
	for j,a in enumerate(row):
		keymap[a]= (i,j)

def keyevent_note(e):
	down= e.type==KEYDOWN
	if e.key in keymap:
		note_state[keymap[e.key]]= int(down)

sample_rate = sd.query_devices(sd.default.device if DUPLEX else sd.default.device[0], 'input')['default_samplerate']
fftlen= fftpack.next_fast_len(fftsize_calc(sample_rate))
#1:1 size of samples:fft
# otherwise rescaling is required, which is pessimum
samples= zeros(fftlen)
fft= zeros(fftlen)
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

	o= outdata[:,0].view()
	i=  indata[:,0].view()

	t1= t0+float(frames)/sample_rate
	if DEBUG:
		print('interval '+str(t0-t1))

	#insert after here
	#DO NOT high latency ops such as allocation, because underflow
	#all bulk ops should be as o[:]=... to prevent realloc
	#todo theres a few copy ops which may or may not need eliminated
	#i believe double buffering wouldnt aid much



	o[:]= linspace(t0, t1, frames, endpoint=False)
	if DEBUG:
		print('frames '+str(frames))
		print('sample_rate '+str(sample_rate))
	#b= push_samples(a)

	#amplitude
	#freq= resample(freq, int(freq.size*2.))
	#freq= freq[:max(1600,fftsize)] #denoise
	#freq= fft(freq, n=fftsize)
	#a= ifft(roll(freq,200.), n=fftsize)
	##a= irfft(a[::], n=frames) ????


	#frequenz
	#todo window
	ifreq= rfft(i)
	#freq= zeros(frames)
	visin= (i.copy(),ifreq.copy())
	#input immutable after here


	#o[:]= sin(o*TAU*60.)
	#o[:]= 1
	##o[:]= irfft(freq)[0:o.size]*8

	#o[:]= sign(a)*abs(a*a*a)

	ofreq= zeros(frames//2)#size will need changed per window
	notes= note_state.flatten()
	notes*= arange(1,1+notes.size)
	notes= pow(notes,sqrt(2))*.5
	notes= notes.astype('int')
	notes= minimum(notes,ofreq.size-1)#clamp index
	ofreq[notes]= 1
	
	#normalize
	s= sum(ofreq)
	if absolute(s)>0.:
		ofreq*= ofreq.size/s

	#lowpass
	_lk= 20#half-frequency
	ofreq*= square(_lk/arange(_lk,_lk+ofreq.size)) # 1 -> lim 0

	o[:]= irfft(ofreq*1j, n=len(o))# *1j does cosine->sine, to taper ends and elim need for windowing
	#o[0]= 

	outmod_postfft= False
	#are operations applied to the output after assigning it from fft inverse
	#if so, fft of output must be recalculated after these operations
	#remember that human hearing is phase-agnostic
	if outmod_postfft:
		None
		ofreq= rfft(o)

	#o[:]+= i
	visout= (o.copy(),ofreq.copy())
	#all transforms after here are not shown on graphs

	#temp window function
	#evited by using sine
	# has slight low freq emission but prevents all popping
	#o[:]*= pow( linspace(0,2,o.size)*linspace(2,0,o.size), 2.)

	if stereo:#chanel mirroring
		outdata[:,1]= o.copy()

	vis.fifo.put_nowait((*visin, *visout))
	#copy is unevitable since o==outdata are managed by outer scope
	#	here was determined to be the appropriate location to copy
	#	manually buffering would still require a copy
	t0= t1
	frame+= frames







def update():
	for e in pygame.event.get():
		if e.type==MOUSEMOTION or e.type==TEXTINPUT:
			continue
		print(e)
		if (e.type == QUIT) or (e.type == KEYUP and e.key == K_ESCAPE):
			pygame.quit()
			return False
		if e.type==KEYDOWN or e.type==KEYUP:
			keyevent_note(e)
	vis.update()
	return True

try:
	with sd.Stream(
		#samplerate=sample_rate,
		#samplerate=44000,
		#blocksize=fftsize,
		#dtype=None, latency=None,
		clip_off=True,
		dither_off= True,
		##never_drop_input= True,
		channels=1,#mono input and output
		#latency='high',
		callback=audio_callback):
			while update(): None;
except Exception as e: 
	print("\nEXCEPT")
    #always check if devices are configured via sd.default.device
	raise e
