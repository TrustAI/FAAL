# +
import cvxpy as cp
import torch
import numpy as np
from torch.autograd import Variable
import torchattacks
import warnings
import mosek
import os
import torch.nn as nn
warnings.filterwarnings("ignore")
os.environ['MOSEKLM_LICENSE_FILE'] = "mosek.lic"

class DAW:

    def __init__(self,
                 train_batch_size,
                 r_choice,
                 learning_approach = "kl",
                 output_return = "weight"
                 ):

        self.train_batch_size = train_batch_size
        self.numerical_eps = 0.000001
        self.learning_approach = learning_approach
        self.output_return = output_return  
        self.r_choice = r_choice
            
      
    
 
    def _initialise_FAAL_problem(self,y):

        # The primal - inner maximisation problem.
        N = self.train_batch_size
        N2= len(y)

        Pemp = 1/N * np.ones(N)  # Change for a diffrent Pemp

        # # Parameter controlling robustness to misspecification
        # α = cp.Constant(self.a_choice)
        # Parameter controlling robustness to statistical error
        r = cp.Constant(self.r_choice)

        # Primal variables and constraints, indep o_initialise_HD_problem2f problem
        self.p = cp.Variable(shape=N, nonneg=True)


        self.nn_loss = cp.Parameter(shape=N)
        self.nn_loss.value = [1/N]*N  # Initialising

        self.worst = cp.Parameter()
        self.worst.value = 0.1  # Initialising

        y = y.cpu()

        # Objective function
        aa = cp.multiply(self.p[0:N][y], self.nn_loss[y])

        objective = cp.Maximize(cp.sum(aa))

        # Simplex constraints
        simplex_constraints = [cp.sum(self.p) == 1]

        kl_constraint = [cp.sum(cp.kl_div(Pemp, self.p)) <= r]
        # t = cp.Variable(name="t", shape=N)
        # kl_constraint = [cp.sum(t) <= r, cp.constraints.exponential.ExpCone(-1*t, Pemp, self.p)]
        
    
        complete_constraints = simplex_constraints + kl_constraint


        # Problem definition
        self.model = cp.Problem(
            objective=objective,
            constraints=complete_constraints)

    
    

   

    
    def solve_weight(self,  y,inf_loss = None, device='cuda'):
        
        '''Solving the primal problem.
           Returning the weighted loss as a tensor Pytorch can autodiff'''

        
        if self.learning_approach == "kl":

            self._initialise_FAAL_problem(y)

        else:
            assert 0 
            


        

        if self.r_choice > 0:
            
            if self.output_return == 'weights':
                self.nn_loss.value = np.array(inf_loss.cpu().detach().numpy())
            
            self.worst.value = np.max(self.nn_loss.value) # DPP step
            
            
            try:
                self.model.solve(solver=cp.ECOS) 
                # ECOS is normally faster than MOSEK for conic problems (it is built for this purpose),
                # but generally also more unstable. 
                # We will revert to MOSEK incase of solving issues.
                # This should happen very infrequently (<1/1000 calls or so, depending on α, r)
                
            except:
                
                try:
                    # self.nn_loss.value += self.numerical_eps # Small amt of noise incase its a numerical issue
                    self.worst.value = np.max(self.nn_loss.value) # Must also re-instate worst-case for DPP
                    self.model.solve(solver=cp.MOSEK)
                    # MOSEK is the second fastest,
                    # But also occasionally fails when α and r are too large.
                
                except:
                    self.model.solve(solver=cp.SCS)
                    # Last resort. Rarely needed.
 

            weights = Variable(torch.from_numpy(self.p.value),
                               requires_grad=True).to(device).float() # Converting primal weights to tensors
            
            
        
            if self.output_return == 'weights':
                
                return weights.detach()
            
            else:
                raise Exception("Not a valid choice of output, please pass pytorch_loss_function if using Pytorch or weights if using another framework")

        else: # If we use only epsilon (could be zero or not)
            
            if self.output_return == 'weights':
                return torch.ones((self.train_batch_size)).cuda()*1/(self.train_batch_size)