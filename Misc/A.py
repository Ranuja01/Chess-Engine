# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 22:17:45 2024

@author: Kumodth
"""

import B

class A():
    
    def __init__(self,a,b,c):
        self.a = a
        self.b = b
        self.c = c


    def addAB(self):
        return self.a + self.b
    
    def test(self):
        B.test1(self)

if __name__ == "__main__":
    a = A(1,2,3)
    
    print(a.addAB())
    
    print(B.addBC(a))
    
    a.test()
    
    print(a.addAB())