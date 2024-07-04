cimport GUI


# Example function using the Pieces class
cdef test_pieces_class():
    # Create an instance of Pieces
    cdef GUI.Pieces piece_instance = GUI("Pawn", "Black", 0, 0)
    
    # Access attributes and print values
    print(piece_instance.piece)
    print(piece_instance.colour)
    print(piece_instance.xLocation)
    print(piece_instance.yLocation)
    print(piece_instance.value)