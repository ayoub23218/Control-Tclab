import numpy as np

import matplotlib.pyplot as plt
from IPython.display import display, clear_output


def LL_RT(MV, Kp, Tlead, Tlag, Ts, PV, PVInit=0, method='EBD'):
    if len(PV) == 0:
        PV.append(PVInit)
        return

    if (Tlag != 0):
        x_k_plus_1 = MV[-1] 
        x_k = MV[-2] 
        y_k = PV[-1]    

        if method == 'EBD':
            den = Tlag + Ts
            PV.append((Tlag/den)*y_k + (Kp*(Tlead+Ts)/den)*x_k_plus_1 - (Kp*Tlead/den)*x_k)
            
        elif method == 'EFD':
            
            PV.append((1 - Ts/Tlag)*y_k + (Kp*Ts/Tlag)*x_k + (Kp*Tlead/Tlag)*(x_k_plus_1 - x_k))
            
    else:
        PV.append(Kp*MV[-1])
        
def PID_RT(SP, PV, Man, MVMan, MVFF, Kc, Ti, Td, alpha, Ts, MVMin, MVMax, MV, MVP, MVI, MVD, E, ManFF = False, PVInit = 0, method = 'EBD'):
    if len(PV) == 0:
        E.append(SP[-1] - PVInit)
    else:
        E.append(SP[-1] - PV[-1])
    # Proportional    
    MVP.append(Kc*E[-1])
    
    # Integral
    if len(MVI) == 0:
        MVI.append((Kc*Ts/Ti)*E[-1])
    else:
        if method == 'TRAP':
            MVI.append(MVI[-1] + (0.5*Kc*Ts/Ti)*(E[-1]+E[-2]))
        else:
            MVI.append(MVI[-1] + (Kc*Ts/Ti)*E[-1])
            
    # Derivative
    if len(MVD) == 0:
        MVD.append((Kc*Td)/(Td*alpha+Ts)*E[-1])
    else:
        if method == 'TRAP':
            MVD.append((((alpha*Td - Ts*0.5) / ((Td*alpha) + Ts*0.5))*MVD[-1]) + (0.5*(Kc*Td)/(Td*alpha+Ts*0.5))*(E[-1] - E[-2]))
        else:
            MVD.append(((alpha*Td / ((Td*alpha) + Ts))*MVD[-1]) +( (Kc*Td)/(Td*alpha+Ts)*(E[-1]-E[-2])))

    # mode manuelle
    if Man[-1] == True:
        if ManFF:
            MVI[-1] = MVMan[-1] - MVP[-1] - MVD[-1] 
        else:
            MVI[-1] = MVMan[-1] - MVP[-1] - MVD[-1] - MVFF[-1]
                
    
   # Saturation
    if (MVP[-1] + MVI[-1] + MVD[-1] + MVFF[-1]) > MVMax:
        MVI[-1] = MVMax - MVP[-1] - MVD[-1] - MVFF[-1]
    if (MVP[-1] + MVI[-1] + MVD[-1] + MVFF[-1]) < MVMin:
        MVI[-1] = MVMin - MVP[-1] - MVD[-1] - MVFF[-1]
            
    # final MV       
    MV.append(MVP[-1] + MVI[-1] + MVD[-1] + MVFF[-1])
    

def IMC_Tuning(Kp, T1, theta, T2=0, tauc=None, model='FOPDT'):
    
    if tauc is None:
        tauc = max(0.8 * T1, 1.5 * theta)
    
    if model == 'FOPDT':

        Kc = (1/Kp) * (T1 / (tauc + theta))
        Ti = T1
        Td = 0 
        
    elif model == 'SOPDT':

        d = (tauc + theta)
        
        Kc = (1/Kp) * (T1 + T2) / d
        Ti = T1 + T2
        Td = (T1 * T2) / (T1 + T2) if (T1 + T2) != 0 else 0
        
    return Kc, Ti, Td
    