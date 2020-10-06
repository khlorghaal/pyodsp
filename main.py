from com import *
import vis

#sounddevice.querydevices()
device_out= 4
device_in= 1

import sounddevice as sd

from numpy import *
from numpy.fft import *
import scipy.fftpack as fftpack

from scipy.interpolate import splrep,splev
from scipy.signal import resample,resample_poly
from scipy.signal.windows import hann

import time
import pygame
import pygame.key
from pygame.locals import *


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

sample_rate = sd.query_devices(device_in, 'input')['default_samplerate']
sample_count= fftpack.next_fast_len(int(sample_rate/FREQ_MINIMUM))
samples= zeros(sample_count)
def push_samples(arr):
	global samples
	o= arr.size
	assert(o<sample_count)
	samples[o:-1]= samples[0:-o-1]
	samples[0:o]= arr
	return samples#*hann(sample_count)*800

def hz_idx(t):
	return (t*sample_count/sample_rate).astype(int)
frame= 0
t0= 0
def audio_callback(indata, outdata, frames, time, status):
	if status:
		print(status)
	global frame
	global data_p
	global data_pp
	global t0

	t1= t0+float(frames)/sample_rate
	##print(t0-t1)
	t= linspace(t0, t1, frames, endpoint=False)
	#print(" "+str(t[0])+" "+str(t[-1]))
	#print(frames)
	#print(sample_rate)
	#inyes= False #this variable has a retarded name so its definitely hack
	#if inyes:
		#if indata.shape[1]==2: indata= indata[:,0]+indata[:,1]	#use separate invocations?
		#a= indata[:,0]
		#b= push_samples(a)

		#amplitude
		#freq= resample(freq, int(freq.size*2.))
		#freq= freq[:max(1600,fftsize)] #denoise
		#freq= fft(freq, n=fftsize)
		#a= ifft(roll(freq,200.), n=fftsize)
	##a= irfft(a[::], n=frames) ????

	#frequenz
	#freq= rfft(b)
	freq= zeros(frames)

	#print(note_array.shape)
	notes= note_state[0]*20
	#freq[hz_idx(notes)]= 1.

	o= outdata[:,0].view()
	#outdata[:,1]= o.view()#mirror channels

	o[:]= sin(t*6.28*60.)
	##o[:]= irfft(freq)[0:o.size]*8

	#o[:]= sign(a)*abs(a*a*a)

	vis.feed_data(o,freq)
	#ownership change! copy if mutating past here
	o= o.copy()
	#freq= freq.copy()

	o*= 0.

	t0= t1
	frame+= frames







def update():
	for e in pygame.event.get():
		if e.type==MOUSEMOTION:
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
	with sd.Stream(device=
		(device_in,device_out),
		#(None,None),
		#sample_rate=None,
		#blocksize=fftsize,
		#dtype=None, latency=None,
		clip_off=True,
		dither_off= True,
		#never_drop_input= True,
		channels=1,#todo? stereo
		latency='high',
		callback=audio_callback):
			while update(): None;
except Exception as e: 
	print("\n")
	raise e #i dont give a fuck
