import pygame
from pygame.locals import *

import ctypes as ct
from OpenGL.GL import *
from OpenGL.GL import shaders

#autoconfigures locations and attribute locations, provides setters
class prog:
	id= -1
	unis= {}
	ulocs= {}
	def bind():
		glUseProgram(self.id)

def prog_vf(vert,frag):
	vert= "#version 450\n"+vert
	frag= "#version 450\n"+frag
	p= prog()
	vert= shaders.compileShader(vert, GL_VERTEX_SHADER)
	frag= shaders.compileShader(frag, GL_FRAGMENT_SHADER)
	p.id= shaders.compileProgram(vert,frag)
	return p

class vbao:
	vbo= -1
	vao= -1
	n= -1
	primitive_type= GL_TRIANGLE_STRIP
	def __init__(self):
		self.vao= glGenVertexArrays(1)
		self.vbo= glGenBuffers(1)
	def bind(self):
		glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
	def draw(self):
		glBindVertexArray(self.vao)
		glDrawArrays(primitive_type, 0, self.n)