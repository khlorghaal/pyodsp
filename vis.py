import com
import pygame
from pygame.locals import *

import ctypes as ct
from OpenGL.GL import *
from OpenGL.GL import shaders

from threading import Lock

from numpy import *

from glutil import *


def resize_pad(arr, l):
    s= arr.size
    d= l-s
    if d>0:
        return concatenate([arr,zeros(d)])
    if d<0:
        return arr[0:l]
    return arr

def genFrameBuffer():
    frame = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, frame)

    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)

    w, h = resolution
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NONE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0)

    assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE)

    return frame, texture

def genTexture_2D_R8(w,h):
    ret= glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D,ret)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R16F, w,h, 0, GL_RED, GL_FLOAT, None)#uninitialized
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    return ret



pygame.init()
resolution = 640,480 #andthussaidtheLourd: 640x480
pygame.display.set_mode(resolution, DOUBLEBUF | OPENGL)
pygame.display.set_caption('____________________________')

freq_min= com.FREQ_MINIMUM
freq_max= com.FREQ_MAXIMUM
WMAX= freq_max

#tex_amp= None
tex_freq= genTexture_2D_R8(WMAX,freq_max)#fixme ehhhh

prog_wave= prog_vf("""
layout(location=0) in vec2 v_p;
void main(){
    vec2 p= v_p;
    //TODO ranging p= pow(sat(1.-abs(uv.y*4.-1.-amp(p.x*1000.))),64.)
    gl_Position = vec4(p, 0,1);
}
""",
"""
out vec4 color;
void main(){
    //todo antialias, maybe
    color= vec4(1);
}
""")
prog_spec= prog_vf("""
layout(location = 0) in vec2 vPos;
layout(location = 1) in vec2 v_uv;
out vec2 uv;
void main(){
    uv= v_uv;
    gl_Position = vec4(vPos, 0,1);
}
""",
"""
uniform sampler2D tex_freq;

float sat(float x){ return min(max(x,0.),1.); }

//cpu manages the ringbuffer uv+quads animation
in vec2 uv;
out vec4 color;
void main(){
    float l= texture(tex_freq, uv).r;
    //l= exp2(l);//TODO colorize
    color= vec4(0,l,0,l);
}
""")

verts= array([
     -1,-1,
     -1, 1,
      1,-1,
      1, 1
    ],
  dtype='float32')


vbao_wave= vbao()
vbao_wave.bind_vbo()
SIZT_WAVE= 2*4
glBufferData(GL_ARRAY_BUFFER, WMAX*SIZT_WAVE, None, GL_DYNAMIC_DRAW)
glBindVertexArray(vbao_wave.vao)
glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)

vbao_spec= vbao()
vbao_spec.bind_vbo()
SIZT_SPEC= 2*4
glBufferData(GL_ARRAY_BUFFER, WMAX*SIZT_SPEC, None, GL_DYNAMIC_DRAW)
glBindVertexArray(vbao_spec.vao)
glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)


clock = pygame.time.Clock()

clean= False
sample_lock= Lock()
_samples_amp= zeros(WMAX)
_samples_freq= zeros(WMAX)

#input audio data
def feed_data(dat_amp,dat_freq):
    assert(dat_amp is not None)
    assert(dat_freq is not None)

    sample_lock.acquire()
    global _samples_amp
    global _samples_freq
    _samples_amp= dat_amp
    _samples_freq= dat_freq
    global clean
    clean= False
    sample_lock.release()

frame=0
#render thread consumption
def update_data():
    sample_lock.acquire()
    global clean
    if clean: return;
    global _samples_amp
    global _samples_freq
    assert(_samples_amp  is not None)
    assert(_samples_freq is not None)
    samples_amp=  _samples_amp.copy()
    samples_freq= _samples_freq.copy()
    sample_lock.release()

    samples_amp= _samples_amp.astype(float32)
    samples_freq= _samples_freq.astype(float32)
    #these are consumed so mutations dont matter

    #amplitude capture
    ##print(samples_amp)
    sample_verts= resize_pad(samples_amp, WMAX)
    assert(sample_verts.size==WMAX)
    sample_verts= array([
        linspace(0,1,WMAX,endpoint=False),#x
        sample_verts#y
        ]).flatten('F').astype(float32)
    vbao_wave.bind_vbo()
    glBufferSubData(GL_ARRAY_BUFFER, 0, sample_verts)

    #freqenz capture into ringbuffer texture
    samples_freq= resize_pad(samples_freq, WMAX)
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, tex_freq)
    global frame
    x0= frame
    w= 1
    h= freq_max
    glTexSubImage2D(GL_TEXTURE_2D, 0, x0,0, w,h, GL_RED, GL_FLOAT, samples_freq)
    frame= (frame+1)%WMAX

    clean= True


def update():
    delta = clock.tick(60)

    #process async bullshit and all buffer updates
    update_data()

    #glBindFramebuffer(GL_FRAMEBUFFER, 0)
    glClearColor(0.0, 0.0, 1.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT)

    #wave
    #fixme uniforms
    prog_wave.bind()
    glUniform1f(uni_time, clock.get_rawtime() / 1000.0)
    vbao_wave.draw()

    #spectrum
    #fixme texture bindings
    prog_spec.bind()
    glActiveTexture(0)
    glBindTexture(GL_TEXTURE_2D, tex_freq)
    vbao_spec.draw()



    pygame.display.flip()
