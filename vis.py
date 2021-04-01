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
resolution= 1920//2,1080//2 #andthussaidtheLourd: 640x480
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


frame=0

class linegraph_fifo:
    #pythonic over gpgpuic
    i= 0
    w= 1024
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
        if d==0: return
        if self.dat.size>amp_arr.size:
            self.dat[d:]= self.dat[:-d]
            self.dat[:d]= amp_arr
        else:
            self.dat= amp_arr[:self.dat.size]
    def set(self, amp_arr):
        self.dat= amp_arr
        w= self.dat.size
        cap= linegraph_fifo.w
        if w>=cap: #over allocation size
            self.dat= self.dat[:cap]
        w= self.dat.size
        self.w= self.n= self.vbao.n= w

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

from queue import Queue
fifo= Queue()# (input amp, input freq, output amp, output freq)
linegraph_i__amp0= linegraph_fifo()
linegraph_i_freq0= linegraph_fifo()
linegraph_o__amp0= linegraph_fifo()
linegraph_o_freq0= linegraph_fifo()
linegraphs= [
    linegraph_i__amp0,
    linegraph_i_freq0,
    linegraph_o__amp0,
    linegraph_o_freq0
]

def update():
    delta = clock.tick(60)

    while not fifo.empty():#CONSUME ALL
        for q,g in zip( fifo.get(), linegraphs ):
            if not isrealobj(q):
                q= absolute(q).real
            #g.push(q.astype(float32))
            g.set(q.astype(float32))

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
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT)

    glEnable(GL_SCISSOR_TEST)
    def vps(W,H):
        W= W/2
        H= H/2
        w= int(floor(W))
        h= int(floor(H))
        W= int(ceil(W))
        H= int(ceil(H))
        return [
            #typing this manually is more readable and writable than permutation
            [0,0,w,h],#bottom left
            [0,h,w,H],#top left
            [w,0,W,h],#bottom right
            [w,h,W,H] #top right
        ]
    viewports= vps(*resolution)
    #viewportactions= [
    #    lambda: (glClearColor(1.0, 0.0, 1.0, 1.0), glClear(GL_COLOR_BUFFER_BIT)),
    #    lambda: (glClearColor(0.0, 0.0, 1.0, 1.0), glClear(GL_COLOR_BUFFER_BIT)),
    #    lambda: (glClearColor(0.0, 1.0, 0.0, 1.0), glClear(GL_COLOR_BUFFER_BIT)),
    #    lambda: (glClearColor(0.0, 1.0, 1.0, 1.0), glClear(GL_COLOR_BUFFER_BIT)) 
    #]
    for v,l in zip(viewports,linegraphs):
        glViewport(*v)
        glScissor(*v)
        l.draw()
    glDisable(GL_SCISSOR_TEST)
    glViewport(0,0,*resolution)

    #spectrum
    #fixme texture bindings
    #prog_spec.bind()
    #glActiveTexture(GL_TEXTURE0)
    #glBindTexture(GL_TEXTURE_2D, tex_freq)
    #vbao_spec.draw()



    pygame.display.flip()
