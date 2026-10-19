# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 01:45:52 2024

@author: Kumodth
"""
import pygame

# Initialize Pygame
pygame.init()

# Set up display
WIDTH, HEIGHT = 800, 800  # Chessboard is 8x8, so make the window square
ROWS, COLS = 8, 8
SQUARE_SIZE = WIDTH // COLS  # Size of each square

# Create the Pygame window
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Chess")

# Define colors
WHITE = (255, 255, 255)
BLACK = (112,128,144)
HIGHLIGHT_COLOR = (0, 255, 0)  # Color to highlight clicked square

# Create a 2D array to store board state (initially empty)
board = [[None for _ in range(COLS)] for _ in range(ROWS)]

def get_square_under_mouse(pos):
    x, y = pos
    row = y // SQUARE_SIZE
    col = x // SQUARE_SIZE
    return row, col

running = True
selected_square = None

def draw_chessboard(win):
    for row in range(ROWS):
        for col in range(COLS):
            color = WHITE if (row + col) % 2 == 0 else BLACK
            pygame.draw.rect(win, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

# Call this function to draw the initial board
draw_chessboard(win)
pygame.display.update()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()
            row, col = get_square_under_mouse(pos)
            selected_square = (row, col)
            
            print(f"Clicked on square: {selected_square}")

    # Highlight the selected square
    if selected_square:
        row, col = selected_square
        pygame.draw.rect(win, HIGHLIGHT_COLOR, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 5)

    pygame.display.update()

pygame.quit()


