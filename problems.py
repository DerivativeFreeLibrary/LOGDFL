import numpy as np

class hs11():
    ''' problema HS11 ma con punto iniziale modificato '''
    n = 2
    m = 1
    x0 = np.array([0., 2.0])
    bl = -np.inf*np.ones(n)
    bu =  np.inf*np.ones(n)
    cl =  np.array([0.0])
    cu =  np.array([1.e+20])
    is_eq_cons = [False]
    def obj(x):
        return (x[0] - 5.0) ** 2 + x[1] ** 2 - 25.0

    def cons(x):
        return np.array([- x[0] ** 2 + x[1]])

class hs12():
    n = 2
    m = 1
    x0 = np.zeros(n)
    bl = -np.inf*np.ones(n)
    bu =  np.inf*np.ones(n)
    cl =  np.array([0.0])
    cu =  np.array([1.e+20])
    is_eq_cons = [False]
    def obj(x):
        return 0.5*x[0]**2 + x[1]**2 - x[0]*x[1] - 7*x[0] - 7*x[1]

    def cons(x):
        return np.array([25.0 -4.0*x[0]**2 - x[1]**2])

class hs14():
    n = 2
    m = 2
    x0 = np.array([2.0,2.0])
    #x0 = np.array([1.0,0.0])
    bl = -np.inf*np.ones(n)
    bu =  np.inf*np.ones(n)
    cl = np.array([-1.0, -1.0])
    cu = np.array([1.e+20, -1.0])
    is_eq_cons = [False,True]
    def obj(x):
        return (x[0]-2.0)**2+(x[1]-1.0)**2
    def cons(x):
        con1 = -0.25*x[0]**2-x[1]**2
        con2 = x[0]-2.0*x[1]
        return np.array([con1,con2])

class hs19():
    n = 2
    m = 2
    x0 = np.array([20.1,5.84])
    bl = np.array([13.0,0.0])
    bu = np.array([100.0,100.0])
    cl = np.array([100.0,-82.81])
    cu = np.array([1.e+20, 1.e+20])
    is_eq_cons = [False,False]
    def obj(x):
        return (x[0]-10)**3+(x[1]-20)**3
    def cons(x):
        con1 = (x[0]-5.0)**2+(x[1]-5.0)**2
        con2 = -(x[0]-6.0)**2-(x[1]-5.0)**2
        return np.array([con1,con2])
