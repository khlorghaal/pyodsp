"""

"""

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



pygame.init()
resolution = 640,480 #andthussaidtheLourd: 640x480
pygame.display.set_mode(resolution, DOUBLEBUF | OPENGL)
pygame.display.set_caption('____________________________')

freq_min= com.FREQ_MINIMUM
freq_max= com.FREQ_MAXIMUM
WMAX= freq_max

#tex_amp= None
tex_freq= genTexture_2D_R8(WMAX,freq_max)#fixme ehhhh

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
    color= vec4(1);//!!
}
""")


#glUniform1d(prog_wave.ulocs["time"], clock.get_rawtime() / 1000.0)

vbao_wave= vbao()
vbao_wave.bind_vbo()
vbao_wave.n= WMAX
SIZT_WAVE= 2*4
glBufferData(GL_ARRAY_BUFFER, vbao_wave.n*SIZT_WAVE, None, GL_DYNAMIC_DRAW)
vbao_wave.bind_vao()
glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)

vbao_spec= vbao()
vbao_spec.bind_vbo()
SIZT_SPEC= 2*4
vbao_spec.n= WMAX
glBufferData(GL_ARRAY_BUFFER, vbao_spec.n*SIZT_SPEC, None, GL_DYNAMIC_DRAW)
vbao_spec.bind_vao()
glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)

glBindVertexArray(0)

fsq= fsq_gen()

clock = pygame.time.Clock()

from queue import Queue
fifo= Queue()

frame=0

class linegraph_fifo:
    #pythonic over gpgpuic
    i= 0
    w= resolution[0]
    prog= prog_vf("""
    layout(location = 0) in vec2 v_p;
    void main(){
        vec2 p= v_p;
        //TODO ranging p= pow(sat(1.-abs(uv.y*4.-1.-amp(p.x*1000.))),64.)
        gl_Position = vec4(p, 0,1);
    }""","""
    layout(location=0) uniform double time;
    out vec4 color;
    void main(){
        //todo antialias, maybe
        color= vec4(1);
    }
    """)
    prog.ulocs["time"]= 0
    def __init__(self):
        self.vbao= vbao()
        self.vbao.primitive_type= GL_LINE_STRIP
        w= self.w
        self.vbao.n= w
        self.dat= zeros(w, dtype='float32')
        self.vbao.bind_vbo()
        glBufferData(GL_ARRAY_BUFFER, w*4*2, None, GL_DYNAMIC_DRAW)
        self.vbao.bind_vao()
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)

    def push(self, amp_arr):
        #idx 0 most recent
        d= amp_arr.size
        if self.dat.size>amp_arr.size:
            self.dat[d:]= self.dat[:-1-d]
            self.dat[:d]= amp_arr
        else:
            self.dat= amp_arr[:self.dat.size]

    def draw(self):
        w= self.w
        assert w==self.dat.size
        v= empty(w*2, dtype='float32')
        v[0::2]= linspace(-1,1,w)
        v[1::2]= self.dat
        #!!v[:]= random.uniform()
        self.vbao.bind_vbo()
        glBufferSubData(GL_ARRAY_BUFFER,0,v)
        self.prog.bind()
        self.vbao.draw()

linegraph_amp0= linegraph_fifo()

def update():
    delta = clock.tick(60)

    while not fifo.empty():#CONSUME ALL
        qp= fifo.get()
        samples_amp=  qp[0].astype(float32)
        samples_freq= qp[1].astype(float32)
        linegraph_amp0.push(samples_amp)
    #amplitude capture
    ##print(samples_amp)
    #sample_verts= resize_pad(samples_amp, WMAX)
    #assert(sample_verts.size==WMAX)
    #sample_verts= array([
    #   linspace(0,1,WMAX,endpoint=False),#x
    #   sample_verts#y
    #   ]).flatten('F').astype(float32)
    #vbao_wave.bind_vbo()
    #glBufferSubData(GL_ARRAY_BUFFER, 0, sample_verts)

    #freqenz capture into ringbuffer texture
    #samples_freq= resize_pad(samples_freq, WMAX)
    #glActiveTexture(GL_TEXTURE0)
    #glBindTexture(GL_TEXTURE_2D, tex_freq)
    #global frame
    #x0= frame
    #w= 1
    #h= freq_max
    #glTexSubImage2D(GL_TEXTURE_2D, 0, x0,0, w,h, GL_RED, GL_FLOAT, samples_freq)
    #frame= (frame+1)%WMAX

    #glBindFramebuffer(GL_FRAMEBUFFER, 0)
    glClearColor(0.0, 0.0, 1.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT)

    linegraph_amp0.draw()
    #spectrum
    #fixme texture bindings
    #prog_spec.bind()
    #glActiveTexture(GL_TEXTURE0)
    #glBindTexture(GL_TEXTURE_2D, tex_freq)
    #vbao_spec.draw()



    pygame.display.flip()
