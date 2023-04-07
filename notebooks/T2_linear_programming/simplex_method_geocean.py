## OPTIMIZATION FUNCTIONS
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd



def Get_New_Tableau(tableau, pivot_column, row_piv_var):
    
    
    new_tableau = tableau.copy()
    print('Old Tableau\n', tableau)
    nr, nc = np.shape(new_tableau)


    #Change pivot_column
    piv_col = np.full(len(tableau.iloc[:, pivot_column]), 0)
    piv_col[row_piv_var] = 1
    new_tableau.iloc[:, pivot_column] = piv_col

    #Multiply row for the value
    piv_r = tableau.iloc[row_piv_var,:]
    new_tableau.iloc[row_piv_var,:] = piv_r / piv_r[pivot_column]

    for c in range(nc):
        for r in range(nr):

            if (c!= pivot_column) & (r!=row_piv_var):

                new_tableau.iloc[r, c] = -tableau.iloc[r, pivot_column] * new_tableau.iloc[row_piv_var, c] + tableau.iloc[r, c] 

    print('\nNew Tableau\n', new_tableau)
    
    return new_tableau



def Check_Optimal(tableau):
    
    print('Last Row is: ' + str(tableau.iloc[-1,:].values))
    
    if np.nanmin(tableau.iloc[-1,:]) >=0:
        print('Optimal solution found')
        optimal_sol = True
    else:
        print('Optimal solution NOT found')

        optimal_sol = False
    


def Get_Optimal_Result(tableau):

    if np.nanmin(tableau.iloc[-1,:]) >=0:
        print('Optimal solution found')
        optimal_sol = True
    else:
        print('Optimal solution NOT found')
        optimal_sol = False
        
    nr, nc = np.shape(tableau)
        
    if optimal_sol:

        #basic variables
        basic_var, optimal_solution, non_basic_var = [], [], []

        for c in range(nc):
            if (len(np.where(tableau.iloc[:,c].values==0)[0])==(tableau.shape[0]-1)) & (len(np.where(tableau.iloc[:,c].values==1)[0])==1): 
                basic_var.append(tableau.keys()[c])
                optimal_solution.append(tableau.iloc[np.where(tableau.iloc[:,c].values==1)[0],-1].values[0])
            else:
                non_basic_var.append(tableau.keys()[c])
                optimal_solution.append(0)
                
        print('Basic Variables: ' +  str(basic_var))
        print('Non Basic Variables: ' +  str(non_basic_var))
                    
        optimal_solution = pd.DataFrame([optimal_solution], columns = tableau.keys())
            
    else:
        print('Optimal solution not found in Tableau')
        
    return optimal_solution
    