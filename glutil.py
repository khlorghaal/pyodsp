import pygame
from pygame.locals import *

import ctypes as ct
from OpenGL.GL import *
from OpenGL.GL import shaders

from numpy import array

#autoconfigures locations and attribute locations, provides setters
class prog:
	id= -1
	unis= {}
	ulocs= {}
	def bind(self):
		glUseProgram(self.id)

class prog_vf(prog):
	def __init__(self, vert,frag):
		super(prog,self).__init__()
		self.vert= "#version 450\n#line 1\n"+vert
		self.frag= "#version 450\n#line 1\n"+frag
		self.ivert= shaders.compileShader(self.vert, GL_VERTEX_SHADER)
		self.ifrag= shaders.compileShader(self.frag, GL_FRAGMENT_SHADER)
		self.id=   shaders.compileProgram(self.ivert,self.ifrag)

class vbao:
	vbo= -1
	vao= -1
	n= -1
	primitive_type= GL_TRIANGLE_STRIP
	def __init__(self):
		self.vao= glGenVertexArrays(1)
		self.vbo= glGenBuffers(1)
	def bind_vbo(self):
		glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
	def bind_vao(self):
		glBindVertexArray(self.vao)
	#def alloc(self, type, size, use='GL_STATIC_DRAW')
	#	bind_vbo()
	#	glbuffer
	#def assign(self, dat):
	#	bind_vbo()
	def draw(self):
		self.bind_vao()
		n= self.n
		assert(n>=0)
		glDrawArrays(self.primitive_type, 0, n)

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


#fullscreen quad
#function generator to allow init-time control
def fsq_gen():
	square= array([
     -1,-1,
     -1, 1,
      1,-1,
      1, 1
    ],
	  dtype='float32')
	fsqv= vbao()
	fsqv.bind_vbo()
	fsqv.n= 4
	glBufferData(GL_ARRAY_BUFFER, fsqv.n*4*2, square, GL_STATIC_DRAW)
	fsqv.bind_vao()
	glEnableVertexAttribArray(0)
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)
	return lambda: fsqv.draw()

