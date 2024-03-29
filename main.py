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

import internal
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
fret_scancodes= array([
		[95,96,97],
		[92,93,94],
		[89,90,91],
		[98,99,88]
	])
note_state= zeros(note_keys.shape)
fret_state= zeros(fret_scancodes.shape)

keymap={}#keycode -> 2d position
for i,row in enumerate(note_keys):
	for j,a in enumerate(row):#a is content of map-cell
		keymap[a]= (i,j)
fretmap={}#scancode -> 2d position
for i,row in enumerate(fret_scancodes):
	for j,a in enumerate(row):
		fretmap[a]= (i,j)

def keyevent_note(e):
	down= e.type==KEYDOWN
	#assert( e.key in keymap != e.scancode in fretmap)
	print(e.key)
	if e.key<9000 and e.key in keymap:#numpad keys have very large keycodes; huge fucking hack
		note_state[keymap[e.key]]= int(down)
	elif e.scancode in fretmap:
		fret_state[fretmap[e.scancode]]= int(down)
	else:
		None#unmapped key




from internal import audio_op
from internal import arity

@audio_op(arity.FFT_OUT)
def piano(rate,in_):
	ofreq= in_
	notes= note_state.flatten()
	notes[:size(fret_state)]+= fret_state.flatten()
	notes*= arange(1,1+notes.size)
	notes= notes*2.
	notes= notes.astype('int')
	notes= minimum(notes,ofreq.size-1)#clamp index
	ofreq[notes]= 1
	#normalize
	s= sum(ofreq)
	if absolute(s)>0.:
		ofreq*= ofreq.size/s
	return ofreq

@audio_op(arity.AMP_OUT)
def synth0(rate,in_):
	return sin(in_*TAU*60./rate)
		#1
		#irfft(freq)[0:o.size]*8
		#sign(a)*abs(a*a*a)

@audio_op(arity.FFT_OUT)
def synth1(rate,f):
	f[2]= 1.
	print(f)
	return f

@audio_op(arity.FFT_INOUT)
def voice(rate,in_):
	#in_[-200::-1]= 0
	#return roll(in_,20)
	s= in_.size
	#f= 200
	#r= 2#int((s+f)/s)
	#b= (s//r)+1
	#in_[:b]= in_[::r]
	#in_[:b]= 0

	#in_= roll(in_,90)

	#f= int(sqrt(sin(in_[0]/600*TAU))*200)
	#in_= roll(in_,f)

	#in_= roll(in_,-30) + roll(in_,-20) + roll(in_,20)*.5 + roll(in_,30)*.25
	#in_/= 3

	#f= lambda i: roll(in_)
	#in_= gather([ f(i) for i in [] ])

	#low
	#in_[:s//2+1]= in_[::2]

	#high
	##in_[:]= in_[::2]*.01

	return in_*.2

#audio_op= piano
#audio_op= synth0
#audio_op= synth1
audio_op= voice

infile= 'voice0.flac'
outfile= 'ses.flac'


internal.invoke(audio_op,infile,outfile)
while True:
	for e in pygame.event.get():
		if e.type==MOUSEMOTION or e.type==TEXTINPUT:
			continue
		#print(e)
		if e.type==KEYDOWN or e.type==KEYUP:
			keyevent_note(e)
		if (e.type == QUIT) or (e.type == KEYUP and e.key == K_ESCAPE):
			internal.quit()
			pygame.quit()
			break
	vis.update()
vis.quit()
internal.quit()