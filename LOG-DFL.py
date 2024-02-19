import numpy as np
import math
import time
import random
from problems import *

import sys

pyc_available = False

class LOG_DFL():

    def __init__(self, funct, n=2, m=0, p=0, gamma=1e-4, delta=0.5, theta1 = 0.5, theta2=0.5, rho = 1.0, eps = np.array([1.0]), q = 1.1):

        '''
        :param n:          Number of variables of the problem ## INT #
        :function funct:   Takes as an input an array(n, ) and returns the value of the objective function ## FUNCTION #
        :param m:          Number of constraints of the problem !!!inequalities + 2*equalities!!! ## INT #
        :param p:          Number of equality constraints of the problem ## INT #
        :param q:          Smoothing exponent of the external penalization ## FLOAT>(1.0,2.0] #
        :param gamma:      Sufficient decrease factor for success in Linesearch ## FLOAT>(0.0,1.0] #
        :param delta:      Dividing factor to expand the steplength ## FLOAT>(0.0,1.0) #
        :param theta1:     Steplength Deacresing factor for successful iterations ## FLOAT>(0.0,1.0)
        :param theta2:     Steplength Deacresing factor for unsuccessful iterations ## FLOAT>(0.0,1.0)
        :param rho:        Internal penalty parameter ## ARRAY(m, ) #
        :param eps:        External penalty parameter ## ARRAY(m, ) #
        :param nf:         Function Evaluations counter ## INT #
        :param gmin:       Minimum(absolute-wise) constraint value found, works as a control parameter ## FLOAT #
        :param alfa_max:   Maximum stored steplength value at current iteration ## FLOAT #
        :param slacks:     Mask to identify which constraints are handled with
                           internal or external penalization ## ARRAY(m, )>(True/False) #
        :param interior:   Saves the presence and the number of constraints with
                           internal penalization ## ARRAY(2, ):[T/F, int] where 0<=int<= m #
        :param factor:
        :param fstop:
        :param xfstop:
        :param alfa_d:
        :param d:
        :param x_localpath:
        '''

        self.F = []
        self.n = n
        self.funct = funct
        self.m = m
        self.p = p
        self.q = q
        self.gamma = gamma
        self.delta = delta
        self.theta1 = theta1  #reduction for success
        self.theta2 = theta2  #reduction for failure
        self.rho = rho
        self.eps = eps
        self.nf = 0
        self.gmin = 0                    #valore minimo dei vincoli
        self.alfa_max = 0.0              #valore massimo degli alfa
        self.slacks = np.array([])
        self.interior = [False, 0]
        self.fstop = []
        self.xfstop= []
        self.alfa_d = np.array([])
        self.d = np.array([])
        self.x_localpath = np.array([])
        self.min_alfamax = np.array([np.inf, 0])
        self.ax = 0.0

    def fpen0(self, x):

        '''
        !!! RETURNS BOTH FUNCTION AND CONSTRAINTS VALUES !!!
        '''
        fob, cin = self.funct(x)
        print(cin)

        if np.any(cin >= -1e-12):  #inequality form: g(x)<=0 --> check if there's a violated constraint
                return np.inf, fob, cin

        f = fob - self.rho * np.sum(np.log(-cin))
        return f, fob, cin

    def flog(self, x):
        '''
        !!! RETURNS ONLY FUNCTION VALUE !!!
        '''
        fob, cin = self.funct(x)
        cin = np.maximum(cin,np.array([-1.e4]*self.m))
        if np.any(cin[np.logical_not(self.slacks)] >= -1e-12):
            return 1.e+16

        f, cin = self.fpen(x)
        return f


    def fpen(self, x, nplus1=False):

        '''
        !!! RETURNS BOTH FUNCTION AND CONSTRAINTS VALUES !!!
        '''
        fob, cin = self.funct(x)  #function and constraints values
        cin = np.maximum(cin, np.array([-1.e4] * self.m))
        nvin = len(cin)           #number of constraints

        if nvin > 0:
            viol = np.sum(np.where(cin<0,0,cin))

            if viol > 1.e-4:
                (self.F).append(np.inf)
            else:
                (self.F).append(fob)
        else:
            (self.F).append(fob)

        if np.any(cin[np.logical_not(self.slacks)] >= -1e-14):
            return np.inf, cin

        if self.interior[0]:  #if there's a constraint handled with internal penalization
            '''
            :(np.minimum)**2:  -->  to leave the "strongly" inner feasible set unperturbed [changed "-" to "+" because of the square power]
            :constant:         -->  to increase the diverging speed of the logarithmic barrier approaching the boundary
            :100.0:            -->  same as "constant", but has more effect when g(x) is close to -1
            :-5:               -->  TO BE EXPLAINED IN A PROPER WAY
            '''
            constant = -(1 / (cin[np.logical_not(self.slacks)] - 10 ** 0))
            constant = 1.0

            f = fob
            for i in range(self.m):
                if not self.slacks[i]:
                    if True: #self.rho> 1.e-4
                        #f += np.minimum(100 * self.rho[i] * constant * (np.log(-cin[i]) - 11.0), 0.0) ** 2
                        f-= 100.0 * self.rho[i] * constant * np.log(-cin[i])
                else:
                    f += 1.0*(np.maximum(0.0, cin[i]) ** self.q) / self.eps[i]

        else:   #only external penalization
            f = fob + np.sum(np.divide(np.maximum(0,cin[self.slacks]) ** self.q, self.eps[self.slacks]))

        if nplus1:  #for n+1 direction
            '''
            The following operation stores information to be used in the n+1 direction exploration.
            Since it's not done anytime the function is called, the "nplus1" parameter is set False by default
            and is set explicitly True whenever this information needs to be stored.
            '''
            if len(self.fstop) < 2 * self.n:
                self.fstop.append(f)
                self.xfstop.append(x)
            else:
                self.fstop.append(f)
                self.xfstop.append(x)
                self.fstop = self.fstop[1:]
                self.xfstop = self.xfstop[1:]

        return f, cin

    def get_constraints(self, x):
        cin_real = get_constraints(x)[np.logical_not(self.slacks)]
        filter_cin = np.maximum(np.array([-1.e4]*cin_real.shape[0]),cin_real)
        return filter_cin



    def dflogbar(self, x, lb, ub, maxfun, tol, iprint):

        '''
        :param x:          Starting point ## ARRAY(self.n, )>FLOAT #
        :param lb:         Lower bound for each variable  ## ARRAY(self.n, )>FLOAT #
        :param ub:         Upper bound for each variable  ## ARRAY(self.n, )>FLOAT #
        :param maxfun:     Maximum number of function evaluations allowed ## INT #
        :param tol:        Tolerance to enter accept stop criterion ## FLOAT #
        :return:
        '''

        ############################################
        ###  FIRST FUNCTION AND CONSTRAINTS EVALUTAION
        ############################################
        f0, fob0, cin0 = self.fpen0(x)
        print('FIRST PASSED', self.n)
        self.nf = 1

        ######################################
        ###  INITIALIZATION
        ######################################
        self.x_centralpath = np.copy(x)
        self.m = len(cin0)
        self.slacks = np.array([False]*self.m)  #set constraints to be all handled by internal penalization
        self.gmin = np.array([0]*self.m)        #initialize gmin to 0(gmin is stored with negative value)
        self.eps = np.array([np.minimum(1/np.abs(fob0),1.e-1)] * self.m)     #initialize external penalty parameters to 0.1
        self.rho = np.array([1.e-1] * self.m)   #initialize internal penalty parameters to 0.1

        ######################################
        ###  CONSTRAINTS SETTING INITIALIZATION
        ######################################
        if f0 == np.inf:
            print("Initial point is not in the strict interior of the feasible region")
            for j in range(self.m):
                if j >= self.m-2*self.p:
                    '''
                    The p couples of inequality constraints(and so 2p constraints) relating to the p equality constraints
                    are stored at the end of the constraints array.
                    These constraints have to be handled by means of EXTERNAL penalization approach.
                    '''
                    self.slacks[j] = True

                elif cin0[j] >= -1e-12:  #if the inequality constraint is violated
                    self.slacks[j] = True  #constraint "j" to be handled by EXTERNAL penalization

                    if np.maximum(cin0[j],0.0) < 1.0:  #if we are close to the feasible region decrease the EXTERNAL penalty parameter
                        self.eps[j] = 1.e-3

                '''elif -cin0[j]!=1:   #constraint handled by INTERNAL penalization
                    # self.rho[j] = np.minimum(0.1 * self.rho[j], -1.e-2 * cin0[j])
                    self.rho[j] = np.minimum(1.0 / np.abs(np.log(-cin0[j])), 1.e-1)  #TO BE EXPLAINED PROPERLY
                    # self.rho[j] = np.minimum(self.factor / np.abs(-cin0[j]), 1.e-1)'''
        else:
            print("Initial point is in the strict interior of the feasible region")
            for j in range(self.m):
                if -cin0[j] != 1:
                    # self.rho[j] = np.minimum(0.1 * self.rho[j], -1.e-2 * cin0[j])
                    #self.rho[j] = np.minimum(1.0 / np.abs(np.log(-cin0[j])), 1.e-1)
                    # self.rho[j] = np.minimum(self.factor / np.abs(-cin0[j]), 1.e-1)
                    break


        if False in self.slacks:
                    '''
                    :self.interior[0]:  -->  FALSE if only external penalization is used, TRUE otherwise
                    :self.interior[1]:  -->  number of constraints with internal penalization(0 if self.interior[0] is FALSE)
                    '''
                    self.interior[0] = True
                    self.interior[1] = np.count_nonzero(self.slacks)   #NOT USED THROGHOUT

        ################################################
        ###  ALGORITHM CONTROL PARAMETERS INITIALIZATION
        ################################################
        self.gmin = -np.array([np.max(-cin0)]*self.m) #first update of gmin
        self.d = np.ones(self.n)  #diagonal of directions matrix
        #self.alfa_d = np.minimum((ub - lb) / 2.0, 1.e+2)
        self.alfa_d = np.maximum(np.minimum(1.0,np.abs(x)), 1.e-3)  #setting stepsizes
        self.alfa_d = np.ones(self.n)
        self.alfa_max = np.max(self.alfa_d)

        ############################################
        ###  FIRST BARRIER FUNCTION EVALUATION
        ############################################
        z = np.copy(x)
        fz, cinz = self.fpen(z, nplus1=True)
        f = fz

        ######################################
        ###  INITAL PRINT
        ######################################
        if (iprint >= 1):
            print(' ----------------------------------\n')
            print(' alfa_max = %f' % self.alfa_max)
            print(' finiz = %e' % f)
            for i in range(self.n):
                print(' xiniz(%d) = %e' % (i, x[i]))

        while True:
            #input()

            ######################################
            ###  PRINT
            ######################################
            if (iprint >= 0):
                if True not in self.slacks:
                    print(' nf = %d  fpen = %.16f f = %e alfamax = %e gmin = %e rho_min = %e  rho_max = %e maxg = %e' % (self.nf, f, self.funct(x)[0],
                                                                                                         self.alfa_max, np.max(self.gmin[np.logical_not(self.slacks)]), np.min(self.rho[np.logical_not(self.slacks)]),
                                                                                                         np.max(self.rho[np.logical_not(self.slacks)]),  np.max(get_constraints(x))))
                elif False not in self.slacks:
                    print(' nf = %d  fpen = %.16f f = %e alfamax = %e gmin = %e eps_min = %e  eps_max = %e maxg = %e' % (self.nf, f, self.funct(x)[0],
                                                                                                         self.alfa_max, np.max(self.gmin), np.min(self.eps[self.slacks]),
                                                                                                                        np.max(self.eps[self.slacks]), np.max(get_constraints(x))))
                else:
                    print(' nf = %d  fpen = %.16f f = %e alfamax = %e gmin = %e rho_min = %e  rho_max = %e eps_min = %e  eps_max = %e maxg = %e' % (self.nf, f, self.funct(x)[0],
                                                                                                         self.alfa_max, np.max(self.gmin[np.logical_not(self.slacks)]), np.min(self.rho[np.logical_not(self.slacks)]),
                                                                                                         np.max(self.rho[np.logical_not(self.slacks)]), np.min(self.eps[self.slacks]),
                                                                                                                        np.max(self.eps[self.slacks]), np.max(get_constraints(x))))
                #print(x)
                #print(self.get_constraints(x))

            if (iprint >= 2):
                for i in range(n):
                    print(' x(%d) = %e' % (i, x[i]))
                    print('max constraints', np.max(cinz) )


            ##############################################
            ###  ENTERING COORDINATE DIRECTIONS ITERATIONS
            ##############################################
            self.x_localpath = np.copy(x)
            #indexes = random.sample(list(np.arange(self.n)), self.n)
            #print(indexes)
            for i in range(self.n):

                ##############################################
                ###  CHECKING STEPLENGTH IS NOT TOO SMALL
                ##############################################
                if (np.abs(self.alfa_d[i]) <= 1.e-3 * np.minimum(1.0, self.alfa_max)):
                    #print('out for small alfa ', i)
                    continue

                if iprint >= 2:
                    print('linesearch over direction', i, self.d[i])
                z, fz = self.Linesearch(i, ub, lb, z, fz, iprint)
                if iprint >= 2:
                    print('after linsearch')
                    print(z, fz)
                '''EXIT FOR i'''

            '''EXIT FOR range(n) per esplorazione delle coordinate'''
            ##############################################
            ###  LOCAL MINIMIZATION PATH DIRECTION
            ##############################################
            self.alfa_max = np.max(self.alfa_d)
            if self.alfa_max >= self.min_alfamax[0]:
                self.min_alfamax[1] += 1
            else:
                self.min_alfamax[0] = np.max(self.alfa_d)
                self.min_alfamax[1] = 0
            if False:#not np.all(z == self.x_localpath)
                z, fz = self.Proj_Linesearch(z, fz, ub, lb, iprint, local_path=True)


            ##############################################
            ###  N+1 DIRECTION
            ##############################################
            z, fz = self.Proj_Linesearch(z = z, fz = fz, ub = ub, lb = lb, coef_gamma = 1.0, iprint = iprint, nplus1=True)
            f = fz

            x = z.copy() #updating x
            self.alfa_max = np.max(self.alfa_d)


           ######################################
            ###  INTERNAL PENALTY PARAMETER UPDATE
            ######################################
            control = 0
            if self.interior[0]: #if internal penalization is used
                '''
                : 1.e-5 : --> increase precision when far from the boundary of the feasible
                              region to better follow the central path
                : everything else : --> theoretical penalty parameter update criterion
                '''
                gmin = np.min(-self.gmin[np.logical_not(self.slacks)])
                if self.rho[0]>=1.e-3:
                    criterion = np.min([1.e-3,self.rho[0]**1.1, gmin**2])
                else:
                    criterion = np.min(
                        [1.e-3, self.rho[0] ** 1.1, gmin ** 2])

                if self.alfa_max <= criterion or (self.min_alfamax[1] >= 50 and self.rho[0]>= 1.e-4)\
                        or (self.min_alfamax[1]>=200 and self.rho[0]>=1.e-6):# or gmin <= 1.e-3 * self.rho[0]:
                    '''
                    : 0.35 : --> normal internal parameter updating speed
                    : square(gmin) : --> increases the updating speed to give more space
                                         to the algorithm when approaching the boundary
                    '''

                    self.rho[0] *= np.minimum(0.35,np.maximum(np.square(np.min(-self.gmin[np.logical_not(self.slacks)])),1.e-2))
                    self.rho = np.array([self.rho[0]]*self.m)
                    '''if 1.e-10<self.rho[0]<=1.e-6 and not(True in self.slacks):
                        self.rho = np.array([1.e-11]*self.m)'''

                    if control == 0: #if you got here the penalty parameter was updated
                        if iprint >= 1:
                            print('updating penalty parameters')
                        self.alfa_d = np.minimum(np.ones(self.n), self.alfa_d * 1.e3)  #update steplengths
                        control = 1

            ######################################
            ###  EXTERNAL PENALTY PARAMETER UPDATE
            ######################################
            if True in self.slacks:  #if external penalization is used
                maxi = np.max(self.eps[self.slacks])   #only update the biggest external penalty parameter
                for j in range(self.m):  #cycling over the constraints
                    if self.slacks[j]:   #if the constraint is externally penalized
                        if self.eps[j] == maxi:  #if the penalty parameter is the maximum one
                            #print('eps',self.eps[j], 'alfamax', self.alfa_max)
                            #input()
                            if self.eps[j] > 1.0e-2*np.sqrt(self.alfa_max):# and self.eps[j]>=self.rho[0]*1.e-4: # and self.nf >= 50 and control < 2:# and self.eps[j]>=self.rho[0]:  #updating criterion
                                #if self.alfa_max <= 1.e-0*self.eps[j]**2:
                                #self.eps = 0.1 * self.eps
                                self.eps[j] = np.minimum(1.e-2*self.eps[j], 1.e-1*np.sqrt(self.alfa_max))  #udpating rule
                                if control == 0:  #if you got here the penalty parameter was updated
                                    self.alfa_d = np.minimum(np.ones(self.n), self.alfa_d * 1.e3)  #update steplengths
                                    control = 2
                                elif control == 1:
                                    control = 2
                if control == 2:
                    control = 1
            ### ALFAMAX UPDATE ###
            self.alfa_max = np.max(self.alfa_d)

            #####################################################
            ###  FUNCTION VALUE UPDATE AND CENTRAL PATH DIRECTION
            #####################################################
            if control == 1:  #penalty parameter updating successful

                self.min_alfamax[0] = np.inf
                self.min_alfamax[1] = 0
                print('updating')
                #print('rho', self.rho[np.logical_not(self.slacks)])
                #print('gmin', self.gmin[np.logical_not(self.slacks)])

                f, cin = self.fpen(x)
                self.nf += 1

                z, fz = self.Proj_Linesearch(z = x, fz = f, ub = ub, lb = lb, coef_gamma = 1.e-2, iprint=iprint, central_path=True)

                f = fz
                x = z.copy()

            #####################################################
            ###  SWITCHING FROM EXTERNAL TO INTERNAL PENALIZATION
            #####################################################
            cin_temp = get_constraints(x)
            for j in range(self.m):
                if self.slacks[j] and j < self.m-2*self.p:
                    if  cin_temp[j]<= -1e-12:
                        print('CHANGING PENALTY')
                        self.slacks[j] = False
                        self.rho[j] = np.max(self.rho)
                        self.p -= 1
                        if self.interior[0] == False:
                            self.interior[0] = True
                        self.interior[1] += 1
                        self.alfa_d = np.minimum(np.ones(self.n), self.alfa_d * 1.e3)

            if self.interior[0]:
                self.gmin = get_constraints(x)

            #####################################################
            ###  STOP CRITERION
            #####################################################
            if self.alfa_max <= tol:
                if self.rho[0] > 1.e-15:
                    if False in self.slacks:
                        self.rho = np.array([1.e-15]*self.m)
                    if True in self.slacks:
                        self.eps = np.array([1.e-15]*self.m)
                    self.alfa_d = self.alfa_d = np.maximum(np.minimum(1.0,np.abs(z)), 1.e-3)
                    #self.alfa_max = np.min(self.alfa_d)
                else:
                    print('alfa_max',self.alfa_max,'tolerance', tol)
                    print('gmin', np.min(self.gmin),' ',np.max(self.gmin), 'tolerance', tol)
                    break
            if self.nf >= maxfun:
                print('maximum function evaluations')
                break
            '''END WHILE'''

        #######################
        ###  FINAL PRINT
        #######################
        if (iprint >= 0):
            if True not in self.slacks:
                print(' nf = %d  fpen = %e f = %e alfamax = %e gmin = %e rho_min = %e  rho_max = %e maxg = %e' % (
                self.nf, f, self.funct(x)[0],
                self.alfa_max, np.min(self.gmin), np.min(self.rho[np.logical_not(self.slacks)]),
                np.max(self.rho[np.logical_not(self.slacks)]), np.max(get_constraints(x))))
            elif False not in self.slacks:
                print(' nf = %d  fpen = %e f = %e alfamax = %e gmin = %e eps_min = %e  eps_max = %e maxg = %e' % (
                self.nf, f, self.funct(x)[0],
                self.alfa_max, np.min(self.gmin), np.min(self.eps[self.slacks]),
                np.max(self.eps[self.slacks]), np.max(get_constraints(x))))
            else:
                print(
                    ' nf = %d  fpen = %e f = %e alfamax = %e gmin = %e rho_min = %e  rho_max = %e eps_min = %e  eps_max = %e maxg = %e' % (
                    self.nf, f, self.funct(x)[0],
                    self.alfa_max, np.max(self.gmin[np.logical_not(self.slacks)]), np.min(self.rho[np.logical_not(self.slacks)]),
                    np.max(self.rho[np.logical_not(self.slacks)]), np.min(self.eps[self.slacks]),
                    np.max(self.eps[self.slacks]), np.max(get_constraints(x))))

        print('f', funct(x)[0], 'fpen', self.fpen(x))
        return x

    def Linesearch(self, i, ub, lb, z, fz, iprint):
        '''
        :param i:          direction index         ## INT #
        :param ub:         upper bounds            ## ARRAY(n,) #
        :param lb:         lower bounds            ## ARRAY(n,) #
        :param z:          starting point          ## ARRAY(n,) #
        :param fz:         starting function value ## FLOAT #
        :param iprint:     print ID                ## INT #
        :return:
        '''

        ##################
        ###  PRINT
        ##################
        if (iprint >= 1):
            print(' j = %d    d(j) = %e\n' % (i, self.d[i]))

        #print('working on ', i)
        for way in [1, 2]:

            feas_step = False
            ######################################
            ###  MAXIMUM STEPLENGTH CHECK
            ######################################
            if self.d[i] > 0.0:  #if direction is positive check upper bound
                alfa_bar = ub[i] - z[i]
                if ((self.alfa_d[i] - alfa_bar) < -1.e-14):  #if the stepsize is feasible for the box
                    alfa = np.maximum(1.e-24, self.alfa_d[i])  #making sure the steplength is not too small
                else:
                    alfa = alfa_bar
            else:   #if direction is negative check lower bound
                alfa_bar = z[i] - lb[i]
                if ((self.alfa_d[i] - alfa_bar) < -1.e-14):  #if the stepsize is feasible for the box
                    alfa = np.maximum(1.e-24, self.alfa_d[i])  #making sure the steplength is not too small
                else:
                    alfa = alfa_bar

            ##########################################
            ###  CHECKING STEPLENGTH IS NOT TOO SMALL
            ##########################################
            if not alfa <= 1.e-3 * np.minimum(1.0, self.alfa_max):
                z_new = z.copy()
                z_new[i] = z[i] + alfa * self.d[i]

                #######################################################################
                ###  SEARCHING FOR FEASIBLE STEPLENGTH WITH RESPECT TO "LOG CONSTRAINTS"
                #######################################################################
                if self.interior[0]:
                    while True:
                        self.nf += 1   #to take into account following call to the blackbox
                        fz_temp, cinz_temp = self.fpen(z_new, nplus1=True)
                        if not np.any(cinz_temp[np.logical_not(self.slacks)] >= -1e-14):
                            fz_new, cinz_new = fz_temp, cinz_temp
                            feas_step = True
                            break
                        alfa /= 2
                        if alfa <= 1.e-14:
                            break
                        z_new[i] = z[i] + alfa * self.d[i]

                else:
                    self.nf += 1  # to take into account following call to the blackbox
                    fz_new, cinz_new = self.fpen(z_new, nplus1=True)
                    feas_step = True

                if feas_step: #not alfa <= 1.e-3 * np.minimum(1.0, self.alfa_max):

                    ########################################
                    ###  SUFFICIENT DECREASE AND EXPANSION
                    ########################################
                    if fz_new < fz - self.gamma * (alfa) ** 2:
                        alfa, fz = self.Expansion(z, fz, fz_new, alfa, alfa_bar, i, self.d[i], iprint)
                        self.alfa_d[i] = alfa * self.theta1  #update the steplength
                        break

                    else:  #if we got here a feasible step was found
                        #print('failed')
                        feas_step = True

            ##########################################
            ###  FIRST FAILURE
            ##########################################
            if way == 1:
                self.d[i] *= -1  #change sense

                if (iprint >= 1):
                    print(' direzione opposta per alfa piccolo\n')
                    print(' j = %d    d(j) = %e\n' % (i, self.d[i]))
                    print(' alfa = %e    alfamax = %e\n' % (alfa, self.alfa_max))

            ##########################################
            ###  SECOND FAILURE
            ##########################################
            else:
                self.d[i] *= -1
                if self.interior[0]:
                    ''' if internal penalty is used reduce the stepsize with resepect to the step used
                    we do this because if we use internal penalty the stepsize could be reduced while
                    searching for feasibility'''
                    if feas_step:
                        self.alfa_d[i] = self.theta2*alfa
                    else:
                        self.alfa_d[i] *= self.theta2

                else:
                    self.alfa_d[i] *= self.theta2

                alfa = 0.0 #double failure

        if alfa >= 1.e-14:
            z[i] = z[i] + alfa * self.d[i]

        return z,fz


    def Expansion(self, x, f, f_new, alfa, alfa_bar, i_corr, d, iprint):

        '''
        :param x:             starting point                                ## ARRAY(n,) #
        :param f:             starting function value                       ## FLOAT #
        :param f_new:         function value for the successful step        ## FLOAT #
        :param alfa:          length of the successful step                 ## FLOAT #
        :param alfa_bar:      maximum feasible stepsize for box constraints ## FLOAT #
        :param i_corr:        direction index                               ## INT #
        :param d:             direction                                     ## INT(1/-1) #
        :param iprint:        print ID                                      ## INT #
        :return:
        '''

        alfa_succ = alfa
        z = x.copy()
        f_succ = f_new

        ###  PRINT  ###
        if (iprint >= 2):
            for i in range(n):
                print('beginning expansion')
                print(' z(%d) = %e' % (i, z[i]))

        while True:

            ###  UPDATING STEPSIZE  ###
            alfa_check = np.minimum(alfa_succ/self.delta, alfa_bar)

            z[i_corr] = x[i_corr]+alfa_check*d #update trial point

            ### NEW FUNCTION EVALUATION ###
            f_temp, cin_new = self.fpen(z)
            self.nf += 1  #to take into account preceding call to the blackbox
            if np.any(cin_new[np.logical_not(self.slacks)] >= -1.e-14):  #check feasibility
                return alfa_succ, f_succ
            elif (f_temp <= f-self.gamma*alfa_check): #if feasible and success
                alfa_succ, f_succ = alfa_check, f_temp  #update
                if self.interior[0]:  #update gmin
                    self.gmin[np.logical_not(self.slacks)] = \
                        np.maximum(self.gmin[np.logical_not(self.slacks)], cin_new[np.logical_not(self.slacks)])
                if alfa_check == alfa_bar:  #if we reached the box
                    return alfa_succ, f_succ
            else:  #if feasible and failure
                return alfa_succ,f_succ


    def Proj_Linesearch(self, z, fz, ub, lb, coef_gamma, iprint, local_path = False, nplus1= False, central_path=False):
        '''
        This is linesearch is different since the direction used is not a coordinate one,
        so one has to use a projection over the box constraints to ensure feasibility with respect
        to lower and upper bounds.
        :param z:              starting point                                ## ARRAY(n,) #
        :param fz:             starting function value                       ## FLOAT #
        :param ub:             upper bounds                                  ## ARRAY(n,) #
        :param lb:             lower bound                                   ## ARRAY(n,) #
        :param coef_gamma:     coefficient to reduce the sufficient decrease ## FLOAT(0,1) #
        :param iprint:         iprint ID                                     ## INT #
        :param local_path:     boolean to identify direction to use          ##T/F #
        :param nplus1:         boolean to identify direction to use          ##T/F #
        :param central_path:   boolean to identify direction to use          ##T/F #
        :return:
        '''

        ################
        ### DIRECTION
        ################
        ways = [1,2]
        if nplus1:
            if iprint >= 2:
                print('direzione n+1')
            alfa1, d1 = self.d_nplus1_constructon()
        if local_path:
            if iprint >= 2:
                print('direzione local path')
            alfa1 = 0.1
            d1 = z - self.x_localpath
        if central_path:
            if iprint >= 2:
                print('direzione central path')
            alfa1= 0.1
            d1 = z - self.x_centralpath
            self.x_centralpath = np.copy(z)
            ways = [1]

        ################
        ### LINESEARCH
        ################
        for way in ways:
            '''d1 is the direction and alfa1 is the stepsize '''

            ### SMALL STEPSIZE ###
            z_new = np.maximum(lb, np.minimum(ub, z + alfa1 * d1))

            ### FEASIBLE STEPSIZE ###
            feas_step = False
            if self.interior[0]:
                while True:
                    self.nf += 1  #to take into account following call to the blackbox
                    fz_temp, cinz_temp = self.fpen(z_new, nplus1=True)
                    if not np.any(cinz_temp[np.logical_not(self.slacks)] >= -1e-14):
                        fz_new, cinz_new = fz_temp, cinz_temp
                        feas_step = True
                        break
                    alfa1 /= 2
                    if alfa1 <= 1.e-14:
                        break
                    z_new = np.maximum(lb, np.minimum(ub, z + alfa1 * d1))
            else:
                self.nf += 1  # to take into account following call to the blackbox
                fz_new, cinz_new = self.fpen(z_new, nplus1=True)
                feas_step = True

            if feas_step:

                '''Se la discesa è sufficiente entra nell'espansione'''
                if fz_new < fz - coef_gamma * self.gamma * (alfa1) ** 2:
                    alfa1, fz = self.Proj_Expansion(z, fz, fz_new, alfa1, d1, lb, ub, nplus1, central_path, iprint)
                    break

            ###################
            ### FIRST FAILURE
            ###################
            if way == 1:
                d1 *= -1

                if (iprint >= 1):
                    print(' direzione n+1-esima opposta per fallimento\n')
                    print(' alfa1 = %e \n' % (alfa1))

                if central_path: #exploiting the central path the direction is not changed
                    alfa1 = 0.0

            ###################
            ### SECOND FAILURE
            ###################
            elif nplus1:  #opposite direction is used only for nplus1 direction
                """Secondo fail"""
                if iprint>=1:
                    print('second fail')
                d1 *= -1
                alfa1 = 0.0


            '''EXIT FOR way'''
        if alfa1 >= 1.e-14:
                z = np.maximum(lb, np.minimum(ub, z + alfa1 * d1))

        return z, fz

    def Proj_Expansion(self, x, f, f_new, alfa, d, lb, ub, nplus1, central_path, iprint):
        '''
        :param x:        starting point                          ## ARRAY(n,) #
        :param f:        starting function value                 ## FLOAT #
        :param f_new:    function value for the successful step  ## FLOAT #
        :param alfa:     stepsize                                ## FLOAT #
        :param d:        direction                             !!## ARRAY(n,) #
        :param lb:       lower bounds                            ## ARRAY(n,)#
        :param ub:       upper bounds                            ## ARRAY(n,)#
        :param iprint:   iprint ID                               ## INT #
        :return:
        '''

        alfa_succ = alfa
        z = x.copy()
        f_succ = f_new


        if (iprint >= 2):
            for i in range(n):
                print(' z(%d) = %e' % (i, z[i]))

        while True:

            ### UPDATING STEPSIZE
            alfa_check = alfa_succ/0.7

            z_old = z.copy()
            z = np.maximum(lb,np.minimum(ub,x+alfa_check*d)) #update trial point
            if np.all(z == z_old):
                return alfa_succ, f_succ

            ### NEW FUNCTION EVALUATION ###
            f_temp, cin_new = self.fpen(z)
            self.nf += 1  #to take into account following call to the blackbox
            if np.any(cin_new >= -1.e-14):
                return alfa_succ,f_succ
            elif (f_temp <= f-self.gamma*alfa_check):
                alfa_succ, f_succ = alfa_check, f_temp  #update
                if self.interior[0]:  #update gmin
                    self.gmin[np.logical_not(self.slacks)] = \
                        np.maximum(self.gmin[np.logical_not(self.slacks)], cin_new[np.logical_not(self.slacks)])
            else:
                return alfa_succ, f_succ

    def d_nplus1_constructon(self):
        iminalfa = np.argmin(self.alfa_d)
        imaxalfa = np.argmax(self.alfa_d)
        dalfamin = self.alfa_d[iminalfa]
        dalfamax = self.alfa_d[imaxalfa]
        alfamedio = (dalfamax + dalfamin) / 2.0
        rapalfa = 3.0
        if dalfamax / dalfamin > rapalfa:
            d1 = -self.d
            dnr = np.sqrt(np.double(self.n))
            alfa1 = np.sum(self.alfa_d) / self.n
        else:
            imin = np.argmin(self.fstop)
            fmin = self.fstop[imin]
            imax = np.argmax(self.fstop)
            fmax = self.fstop[imax]
            d1 = self.xfstop[imin] - self.xfstop[imax]
            dnr = np.linalg.norm(d1)
            alfa1 = alfamedio

        return alfa1, d1

###########################################################

def prob_cute_wrap(x):
    f = prob_cute.obj(x)
    c = prob_cute.cons(x)
    ieq = np.array(prob_cute.is_eq_cons)
    cineq = c[np.logical_not(ieq)]
    ceq = c[ieq]
    c_new = np.array([])
    cl = prob_cute.cl[np.logical_not(ieq)]
    cu = prob_cute.cu[np.logical_not(ieq)]
    for i in range(len(cl)):

        if cl[i] > -1e20 and cu[i] < 1e20:
            c_new = np.append(c_new, cl[i]-cineq[i])
            c_new = np.append(c_new, cineq[i]-cu[i])

        elif cu[i] < 1e20:
            c_new = np.append(c_new, cineq[i]-cu[i])

        elif cl[i] > -1e20:
            c_new = np.append(c_new, cl[i]-cineq[i])

    for i in range(len(ceq)):

        c_new = np.append(c_new, ceq[i])
        c_new = np.append(c_new, -ceq[i])

    return f, c_new

def get_constraints(x):

    c = prob_cute.cons(x)
    ieq = np.array(prob_cute.is_eq_cons)
    ceq = c[ieq] - prob_cute.cl[ieq]
    cineq = c[np.logical_not(ieq)]
    c_new = np.array([])
    cl = prob_cute.cl[np.logical_not(ieq)]
    cu = prob_cute.cu[np.logical_not(ieq)]
    for i in range(len(cl)):

        if cl[i] > -1e20 and cu[i] < 1e20:
            c_new = np.append(c_new, cl[i]-cineq[i])
            c_new = np.append(c_new, cineq[i]-cu[i])

        elif cu[i] < 1e20:
            c_new = np.append(c_new, cineq[i]-cu[i])

        elif cl[i] > -1e20:
            c_new = np.append(c_new, cl[i]-cineq[i])

    for i in range(len(ceq)):

        c_new = np.append(c_new, ceq[i])
        c_new = np.append(c_new, -ceq[i])

    return c_new

funct = prob_cute_wrap

maxfun  = 20000
tol     = 1.e-14
iprint  = 0
name    = 'HS11'

print('Problem', name)
prob_name = name
if name == 'HS12':
    prob_cute = hs12
elif name == 'HS11':
    prob_cute = hs11
elif name == 'HS19':
    prob_cute = hs19
elif name == 'HS14':
    prob_cute = hs14
else:
    print('The problem {} has not yet been implemented.'.format(name))

n = prob_cute.n
x0 = prob_cute.x0
lb = prob_cute.bl
ub = prob_cute.bu
print('lb = ',lb)
print('ub = ',ub)
x0 = np.maximum(lb,np.minimum(ub,x0))
print('x0 = ',x0)

ieq = np.array(prob_cute.is_eq_cons)
c = prob_cute.cons(x0)
p = len(c[ieq])

model = LOG_DFL(funct=funct, n=n, p=p, delta=0.5)

f, cin = model.funct(x0)

print('constraints', prob_cute.cons(x0))

x = model.dflogbar(x0, lb, ub, maxfun, tol, iprint)

F = np.asarray(model.F)
print('fbest = ',np.min(F),' nf = ',np.argmin(F))
if len(F)<maxfun:
   F = np.concatenate((F,np.inf*np.ones(maxfun-len(F))),axis=0)

print('\nx finale : ', x)
print('\nFunction, (Constraints) : ', model.funct(x))
