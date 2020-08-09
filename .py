import pygame
from pygame.locals import *

import ctypes as ct
from OpenGL.GL import * 
from OpenGL.GL import shaders

from threading import Lock

from sys import exit as exitsystem

from numpy import *


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

def genTexture_1D_R8(w):
    ret= glGenTextures(1)
    glBindTexture(GL_TEXTURE_1D,ret)
    glTexImage1D(GL_TEXTURE_1D, 0, GL_R32F, w, 0, GL_RED, GL_FLOAT, None)#uninitialized
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    return ret

VERT_SRC = """
#version 450
layout(location = 0) in vec2 vPos;
void main(){ gl_Position = vec4(vPos, 0,1); }
"""

FRAG_SRC = """
#version 450
#define fragCoord gl_FragCoord.xy
uniform vec2  mouse;
uniform float time;
uniform vec2  res;
layout(location = 0) uniform sampler1D tex_amp;
layout(location = 1) uniform sampler1D tex_freq;
out vec4 fragColor;

float  amp(float x){ return texture(tex_amp, x).r; }
float freq(float x){ return texture(tex_freq,x).r; }
void main()
{
    vec2 uv = fragCoord/res.xy;
    vec2 p= uv*2-1;
    p.x*= res.x/res.y;
    vec3 color= vec3(
        (uv.y>.5)? 
            exp2(amp(uv.x)*4.-4.):
            exp2(freq(uv.x)*4.-4.)
        );
    fragColor = vec4(color, 1.0);
}
"""



pygame.init()
resolution = 640,480 #and thus said the lourd, 640x480
pygame.display.set_mode(resolution, DOUBLEBUF | OPENGL)
pygame.display.set_caption('PIRATE?-PyShadeToy')        

vert = shaders.compileShader(VERT_SRC, GL_VERTEX_SHADER)
frag = shaders.compileShader(FRAG_SRC, GL_FRAGMENT_SHADER)

shader = shaders.compileProgram(vert,frag)

uni_mouse= glGetUniformLocation(shader,'mouse')
uni_ticks= glGetUniformLocation(shader,'time')
uni_tex_amp = glGetUniformLocation(shader,'tex_amp' )
uni_tex_freq= glGetUniformLocation(shader,'tex_freq')
uni_res= glGetUniformLocation(shader,'res')

glUseProgram(shader)

verts= array([
     -1,-1,
     -1, 1,
      1,-1,
      1, 1
    ],
  dtype='float32')


vao = glGenVertexArrays(1)
glBindVertexArray(vao)
vbo = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, vbo)
glBufferData(GL_ARRAY_BUFFER, verts, GL_STATIC_DRAW)
glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)

TEX_WMAX= 2**10
tex_amp=  genTexture_1D_R8(TEX_WMAX)
tex_freq= genTexture_1D_R8(TEX_WMAX)

clock = pygame.time.Clock()

clean= False
sample_lock= Lock()
_samples_amp= None
_samples_freq= None
#input audio data
#todo this wont work lol
def feed_data(dat_amp,dat_freq):
    assert(dat_amp is not None)
    assert(dat_freq is not None)
    global clean
    global _samples_amp
    global _samples_freq

    sample_lock.acquire()
    _samples_amp= dat_amp
    _samples_freq= dat_freq
    sample_lock.release()
    clean= False
feed_data(arange(1),arange(1))#init

#render thread consumption
def update_data():
    global clean
    if(clean): return;
    global _samples_amp
    global _samples_freq
    assert(_samples_amp is not None)
    assert(_samples_freq is not None)
    sample_lock.acquire()
    samples_amp= _samples_amp.copy()
    samples_freq= _samples_freq.copy()
    sample_lock.release()
    #discard overflow
    if(samples_amp.size>TEX_WMAX):
        samples_amp=  samples_amp[:TEX_WMAX]
    if(samples_freq.size>TEX_WMAX):
        samples_freq= samples_freq[:TEX_WMAX]
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_1D, tex_amp)
    glTexSubImage1D(GL_TEXTURE_1D, 0,0,  samples_amp.size, GL_RED, GL_FLOAT, samples_amp)
    glActiveTexture(GL_TEXTURE1)
    glBindTexture(GL_TEXTURE_1D, tex_freq)
    glTexSubImage1D(GL_TEXTURE_1D, 0,0, samples_freq.size, GL_RED, GL_FLOAT, samples_freq)
    clean= True


def update():
    delta = clock.tick(60)

    for event in pygame.event.get():
        if (event.type == QUIT) or (event.type == KEYUP and event.key == K_ESCAPE):
            pygame.quit()
            return False

    #process async bullshit
    update_data()

    #glBindFramebuffer(GL_FRAMEBUFFER, 0)
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT)
    
    glUseProgram(shader)
    glUniform2f(uni_mouse, *pygame.mouse.get_pos())
    glUniform1f(uni_ticks, clock.get_rawtime() / 1000.0)
    glUniform1i(uni_tex_amp, 0);
    glUniform1i(uni_tex_freq,1);
    glUniform2f(uni_res, *resolution)
    glBindVertexArray(vao)

    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_1D, tex_amp)
    glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_1D, tex_freq)
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

    
    pygame.display.set_caption("FPS: {}".format(clock.get_fps()))
    pygame.display.flip()

    return True
