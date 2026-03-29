import numpy as np

import matplotlib.pyplot as plt
from IPython.display import display, clear_output


def LL_RT(MV, Kp, Tlead, Tlag, Ts, PV, PVInit=0, method='EBD'):
    
    """
    The function "LL_RT" needs to be included in a "for or while loop".
    
    :MV: MV (or Manipulated Value / Input) vector
    :Kp: process gain
    :Tlead: lead time constant [s]
    :Tlag: lag time constant [s]
    :Ts: sampling period [s]
    
    :PV: PV (or Process Value / Output) vector
    
    :PVInit: Initial value for PV (optional: default value is 0): used if LL_RT is ran first in the sequence and no value of PV is available yet.
    
    :method: discretisation method (optional: default value is 'EBD')
        EBD: Euler Backward Difference
        EFD: Euler Forward Difference
        
    The function "LL_RT" appends new values to the vector "PV".
    The appended values are based on the Lead-Lag transfer function:
    P(s) = Kp * (Tlead*s + 1) / (Tlag*s + 1)
    """
    
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
    
    """ 
    The function "PID_RT" needs to be included in a "for or while loop".
    
    :SP: SP (or SetPoint) vector
    :PV: PV (or Process Value) vector
    :Man: Man (or Manual controller mode) vector (True or False)
    :MVMan: MVMan (or Manual value for MV) vector
    :MVFF: MVFF (or Feedforward) vector
    
    :Kc: controller gain
    :Ti: integral time constant [s]
    :Td: derivative time constant [s]
    :alpha: Tfd = alpha*Td where Tfd is the derivative filter time constant [s]
    :Ts: sampling period [s]
    
    :MVMin: minimum value for MV (used for saturation and anti wind-up)
    :MVMax: maximum value for MV (used for saturation and anti wind-up)
    
    :MV: MV (or Manipulated Value) vector
    :MVP: MVP (or Propotional part of MV) vector
    :MVI: MVI (or Integral part of MV) vector
    :MVD: MVD (or Derivative part of MV) vector
    :E: E (or control Error) vector
    
    :ManFF: Activated FF in manual mode (optional: default boolean value is False)
    :PVInit: Initial value for PV (optional: default value is 0): used if PID_RT is ran first in the sequence and no value of PV is available yet.
    
    :method: discretisation method (optional: default value is 'EBD-EBD')
        EBD-EBD: EBD for integral action and EBD for derivative action
        EBD-TRAP: EBD for integral action and TRAP for derivative action
        TRAP-EBD: TRAP for integral action and EBD for derivative action
        TRAP-TRAP: TRAP for integral action and TRAP for derivative action
        
    The function "PID_RT" appends new values to the vectors "MV", "MVP", "MVI", and "MVD".
    The appended values are based on the PID algorithm, the controller mode, and feedforward.
    Note that saturation of "MV" within the limits [MVMin MVMax] is implemented with anti wind-up.
    """
    
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
    
    """
    The function "IMC_Tuning" calculates the PID controller parameters (Kc, Ti, Td) 
    based on the Internal Model Control (IMC) rules.
    
    :Kp: process gain
    :T1: dominant time constant [s] (Tp1)
    :theta: process dead time [s] (thetap)
    :T2: second time constant [s] (Tp2) - (optional: default value is 0)
    :tauc: desired closed-loop time constant [s] (optional: default calculation provided)
    
    :model: type of process model used for tuning (optional: default is 'FOPDT')
        'FOPDT': First Order Plus Dead Time
        'SOPDT': Second Order Plus Dead Time
        
    :return: 
        Kc: controller gain
        Ti: integral time constant [s]
        Td: derivative time constant [s]
        
    The function uses the simplified IMC rules (Skogestad) where:
    - For FOPDT: P(s) = Kp * exp(-theta*s) / (T1*s + 1)
    - For SOPDT: P(s) = Kp * exp(-theta*s) / ((T1*s + 1)*(T2*s + 1))
    """
    
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


def Margin(Kp, Tp1, Tp2, thetap, Kc, Ti, Td, alpha, omega):
    
    """
    The function "Margin" calculates the stability margins of the closed-loop system
    based on the frequency response of the Open-Loop transfer function L(s) = C(s) * P(s).
    
    :Kp: process gain
    :Tp1: process dominant time constant [s]
    :Tp2: process second time constant [s]
    :thetap: process dead time (delay) [s]
    
    :Kc: controller gain
    :Ti: integral time constant [s]
    :Td: derivative time constant [s]
    :alpha: derivative filter factor
    
    :omega: frequency vector [rad/s] (usually created with np.logspace)
    
    :return:
        GM: Gain Margin (linear scale)
        PM: Phase Margin [degrees]
        wc: Crossover frequency [rad/s] (frequency where gain = 1 or 0 dB)
        w180: Ultimate frequency [rad/s] (frequency where phase = -180°)
        
    The Gain Margin (GM) indicates how much the controller gain can be increased before 
    the system becomes unstable. 
    The Phase Margin (PM) indicates how much additional delay or phase lag the system 
    can tolerate before becoming unstable.
    
    Typical stability criteria: 5 (14 dB) > GM > 1.7 (5 dB) and 60° > PM > 30°.
    """
    s = 1j * omega
    
    P = (Kp * np.exp(-thetap * s)) / ((Tp1 * s + 1) * (Tp2 * s + 1))
    C = Kc * (1 + 1/(Ti * s) + (Td * s) / (alpha * Td * s + 1))
    
    L = P * C
    
    gain = np.abs(L)
    phase = np.unwrap(np.angle(L)) * 180 / np.pi

    idx_wc = np.where(np.diff(np.sign(gain - 1)))[0]
    
    if len(idx_wc) > 0:
        i = idx_wc[0]
        log_w = np.log10(omega[i]) + (np.log10(omega[i+1]) - np.log10(omega[i])) * \
                (0 - 20*np.log10(gain[i])) / (20*np.log10(gain[i+1]) - 20*np.log10(gain[i]))
        wc = 10**log_w
        
        phase_wc = np.interp(wc, omega, phase)
        PM = 180 + phase_wc
    else:
        wc = np.nan
        PM = np.inf

    idx_w180 = np.where(np.diff(np.sign(phase + 180)))[0]
    
    if len(idx_w180) > 0:
        j = idx_w180[0]
        w180 = omega[j] + (omega[j+1] - omega[j]) * \
               (-180 - phase[j]) / (phase[j+1] - phase[j])
        gain_w180 = np.interp(w180, omega, gain)
        GM = 1 / gain_w180
    else:
        w180 = np.nan
        GM = np.inf
    
    return GM, PM, wc, w180

    