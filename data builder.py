import pygame as py
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
from openpyxl import load_workbook
import random
import sys

'''
use a hotkey to reset to new background color
choose black or white text as best option
once you select, it uploads to excel spreadsheet (r,g,b) of background and (0) for white or (1) for black
'''

BLACK = (0,0,0)
WHITE = (255,255,255)

py.init()
window = py.display.set_mode((400,400))
py.display.set_caption('data generator')
font = py.font.Font('freesansbold.ttf', 20)

r,g,b = random.randint(0,255),random.randint(0,255),random.randint(0,255)
color = (r,g,b)
text1 = font.render("BLACK TEXT (1)", True, BLACK, color)
text2 = font.render("WHITE TEXT (0)", True, WHITE, color)

text1Rect = text1.get_rect()
text2Rect = text2.get_rect()

text1Rect.center = (200,150)
text2Rect.center = (200,250)

data = { 'r':[],'g':[],'b':[],'output':[]}

running = True
while running:
	for event in py.event.get():
		if event.type == py.QUIT:
			running = False

		if event.type == py.KEYDOWN:

			if event.key == py.K_1:
				r,g,b = random.randint(0,255),random.randint(0,255),random.randint(0,255)
				text1 = font.render("BLACK TEXT (1)", True, BLACK, (r,g,b))
				text2 = font.render("WHITE TEXT (0)", True, WHITE, (r,g,b))
				data['r'].append(r)
				data['g'].append(g)
				data['b'].append(b)
				data['output'].append(1)

			elif event.key == py.K_0:
				r,g,b = random.randint(0,255),random.randint(0,255),random.randint(0,255)
				text1 = font.render("BLACK TEXT (1)", True, BLACK, (r,g,b))
				text2 = font.render("WHITE TEXT (0)", True, WHITE, (r,g,b))
				data['r'].append(r)
				data['g'].append(g)
				data['b'].append(b)
				data['output'].append(0)


	color = (r,g,b)

	window.fill(color)

	window.blit(text1, text1Rect)
	window.blit(text2, text2Rect)

	py.display.update()

df = pd.DataFrame(data)
writer = ExcelWriter('data.xlsx')
writer.book = load_workbook('data.xlsx')
writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
reader = pd.read_excel(r'data.xlsx')
df.to_excel(writer, index=False, header=False, startrow=len(reader)+1)
writer.close()