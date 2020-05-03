# fork of https://github.com/Blakkis/GLSL_Python/blob/master/multipass_setup.py

import pygame
from pygame.locals import *

import ctypes as ct
from OpenGL.GL import * 
from OpenGL.GL import shaders
#from OpenGL.GLU import *

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
uniform sampler1D tex;
out vec4 fragColor;

float wave(float x){
    return texture(tex,x).r;
}
void main()
{
    vec2 uv = fragCoord/res.xy;
    vec2 p= uv*2-1;
    p.x*= res.x/res.y;
    vec3 color= vec3(wave(uv.x));
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

uni_mouse = glGetUniformLocation(shader, 'mouse')
uni_ticks = glGetUniformLocation(shader, 'time')
uni_tex = glGetUniformLocation(shader, 'tex')

glUseProgram(shader)
glUniform2f(glGetUniformLocation(shader, 'iResolution'), *resolution)

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
glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2*4, None)

TEX_WMAX= 2048
tex= genTexture_1D_R8(TEX_WMAX)

clock = pygame.time.Clock()

unclean= False
samples= None
#input audio data
#todo this wont work lol
def feed_data(dat):#FIXME race conditiony BAD BAD BAD !!!!! >:C
    #print("ASDFADSGAETRHUGARTEHU")
    #print(dat)
    #print(arange(1)+1)

    assert(dat is not None)
    global unclean
    global samples
    samples= dat.copy()
    #fixme this may crash horrible from race condition ¯\_(ツ)_/¯
    unclean= True
feed_data(arange(1))#init

#render thread consumption
def update_data():
    global unclean
    global samples
    assert(samples is not None)
    if(unclean):
        glBindTexture(GL_TEXTURE_1D, tex)
        w= samples.size
        if(w>TEX_WMAX):#discard overflow
            samples= samples[:TEX_WMAX]
        glTexSubImage1D(GL_TEXTURE_1D, 0,0, w, GL_RED, GL_FLOAT, samples)
        unclean= False


def update():
    delta = clock.tick(8192)#todo why this number lol

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
    glUniform1f(uni_ticks, pygame.time.get_ticks() / 1000.0)
    glUniform1i(uni_tex, 0);
    glBindVertexArray(vao)
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

    
    pygame.display.set_caption("FPS: {}".format(clock.get_fps()))
    pygame.display.flip()

    return True
