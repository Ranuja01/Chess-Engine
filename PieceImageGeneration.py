
names = {"blackBishopWhiteSquare","blackQueenBlackSquare",
          "blackBishopBlackSquare","blackKnightBlackSquare",
          "blackKnightWhiteSquare", "blackRookWhiteSquare",
          "blackRookBlackSquare", "blackPawnWhiteSquare",
          "blackPawnBlackSquare","blackKingWhiteSquare",
          "blackKingBlackSquare","blackQueenWhiteSquare",
          
          "whiteBishopWhiteSquare","whiteQueenBlackSquare",
          "whiteBishopBlackSquare","whiteKnightBlackSquare",
          "whiteKnightWhiteSquare", "whiteRookWhiteSquare",
          "whiteRookBlackSquare", "whitePawnWhiteSquare",
          "whitePawnBlackSquare","whiteKingWhiteSquare",
          "whiteKingBlackSquare","whiteQueenWhiteSquare"}

for i in names:
    var1 = i + " = Image.open('" + i + ".png')"
    var2 = i +" = " + i + ".resize((90, 90), Image.ANTIALIAS)"
    var3 = "self.canvas." + i + "Image = ImageTk.PhotoImage(" + i + ",master = self)"
    print (var1)
    print (var2)
    print (var3)
    print ("")
  