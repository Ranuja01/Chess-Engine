cdef class Pieces:
    cdef public str piece
    cdef public str colour
    cdef public int xLocation
    cdef public int yLocation
    cdef public int value

    def __init__(Pieces self, str pieceType, str colour, int x, int y)