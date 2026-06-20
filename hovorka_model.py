import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class HovorkaConstants:
    def __init__(self, BW=70.0, u_basal=12.9127):
        self.BW = BW                        

        # --- Glucose Subsystem (Table 2) ---
        self.F01 = 0.00097 * BW             
        self.Vg = 0.16 * BW                 
        self.k12 = 0.066                    
        self.Mwg = 180.0                    
        self.Gb = 90.0                      
        self.Gth = 162.0                    
        self.Gth1 = 60.0                    
        self.ke1 = 0.007                    
        
        # --- Hepatic EGP Parameters (Table 3) ---
        self.EGP_b = 1.23                   
        self.K6gp = 0.034                   
        self.Ggg1b = 0.7425                 
        self.Ggng1b = 0.495                 
        self.Sc = 297.0                     
        self.Hth = 80.0e-7                  
        self.tD = 59.90                     
        self.tau = 23.24                    
        self.kp2 = 0.0007                   
        self.n = 0.01                       
        self.rho = 0.86                     
        self.Hb = 58.0e-7                   
        self.delta = 0.98e-7                
        self.sigma = 1.714410e-11           
        
        # --- Insulin Subsystem Parameters (Table 4) ---
        self.Vi = 0.12 * BW                 
        self.tau_s = 55.0                   
        self.ke = 0.138                     
        self.k21 = 0.045                    
        self.kd = 0.0021                    
        self.ka = 0.02                      

        # --- Meal Absorption Parameters (Table 2) ---
        self.Ag = 0.8                       
        self.tau_d = 40.0                   

        # --- Insulin Action Parameters (Table 4) ---
        self.ka1 = 0.006                    
        self.ka2 = 0.06                     
        self.ka3 = 0.03                     
        self.kb1 = 3.072e-5                 
        self.kb2 = 4.92e-5                  
        self.kb3 = 0.00156                  

        self.u_basal = u_basal            
        self.u0 = u_basal

        # --- PURE INITIAL CONDITIONS ---
        self.S1_0 = u_basal * self.tau_s    
        self.S2_0 = u_basal * self.tau_s    
        self.I_0 = u_basal / (0.01656 * BW) 
        self.x1_0 = 0.30898 * u_basal / BW  
        self.x2_0 = 0.04951 * u_basal / BW  
        self.x3_0 = 3.2206 * u_basal / BW   
        
        # Exact resting baselines from Mam's tables
        self.G_0 = 90.0                     
        self.Gt_0 = 70.0                    

        self.Dm1_0 = 0.0
        self.Dm2_0 = 0.0
        
        # Perfect mathematical steady state initialization
        self.G6p_0 = (self.Ggg1b + self.Ggng1b) / self.K6gp  
        self.H_0 = self.Hb 
        self.SRH_S_0 = self.n * self.Hb                 

        self.MAX_TIME = 1440              
        self.h = 0.1                        

class HovorkaModel:
    def __init__(self, BW=70.0, u_basal=12.9127):
        self.BW = BW
        self.u_basal = u_basal

    def meal_input(self, t, meal_times, meal_durations, meal_cho):
        d_cho = 0.0
        for i in range(len(meal_times)):
            if meal_times[i] <= t < meal_times[i] + meal_durations[i]:
                d_cho = meal_cho[i] / meal_durations[i]
                break
        return d_cho

    def insulin_input(self, t, bolus_times, bolus_values, bolus_duration=15.0):
        for i in range(len(bolus_times)):
            if bolus_times[i] <= t < bolus_times[i] + bolus_duration:
                return bolus_values[i]
        return self.u_basal

    def odes(self, y, t, meal_times, meal_durations, meal_cho, bolus_times, bolus_values, bolus_duration, c, model_type="proposed"):
        # 13-STATE DIFFERENTIAL MATRIX
        Dm1, Dm2, G, Gt, G6p, H, SRH_S, S1, S2, x1, x2, x3, I = y

        G = max(10.0, G)
        G6p = max(0.0, G6p)
        H = max(0.0, H)

        # --- 1. Meal Absorption ---
        d_cho = self.meal_input(t, meal_times, meal_durations, meal_cho)
        D_meal = 1000.0 * d_cho / c.Mwg
        
        dDm1 = c.Ag * D_meal - Dm1 / c.tau_d
        dDm2 = Dm1 / c.tau_d - Dm2 / c.tau_d
        Ug = Dm2 / c.tau_d

        # --- 2. Glucose Clearance ---
        Ugc = 18.0 * Ug / c.Vg
        F01uc = 18.0 * c.F01 / c.Vg if G >= 81.0 else (18.0 * c.F01 * G) / (c.Vg * 81.0)
        Erc = c.ke1 * (G - c.Gth) if G >= c.Gth else 0.0
        
        B = Ugc - F01uc - Erc + (c.k12 * (Gt - c.Gb)) - (x1 * (G - c.Gb))
        
        # --- 3. Hepatic EGP & Glucagon Dynamics ---
        if model_type == "proposed":
            # Dalla Man 2007 Glucagon Differential Equations
            target_SRH = c.n * c.Hb if G >= c.Gb else max(c.sigma * (c.Gth1 - G) / (I + 1.0) + c.n * c.Hb, 0.0)
            dSRH_S = -c.rho * (SRH_S - target_SRH)
            
            E = (1.0 - np.tanh((t - c.tD) / c.tau)) / 2.0
            Ggg = (c.Ggg1b + c.Sc * max(0.0, H - c.Hth)) * E
            dG6p = -c.K6gp * G6p + Ggg + c.Ggng1b
            
            # EGP in correct mg/dl/min scale
            EGP_base = c.K6gp * G6p - c.kp2 * (G - c.Gb)
            dGdt_pos = (B + EGP_base) / (1.0 + x3)
            
            EGPc = EGP_base - x3 * dGdt_pos if dGdt_pos >= 0 else EGP_base
            
            dG = B + EGPc
            SRH_d = c.delta * max(-dG, 0.0)
            dH = -c.n * H + SRH_S + SRH_d
        else:
            EGPc = c.EGP_b * np.exp(-x3)
            dG6p = 0.0
            dH = 0.0
            dSRH_S = 0.0
            dG = B + EGPc

        # --- 4. System Differentials ---
        dGt = (x1 * (G - c.Gb)) - ((c.k12 + x2) * (Gt - c.Gb))
        
        u = self.insulin_input(t, bolus_times, bolus_values, bolus_duration)
        dS1 = u - S1 / c.tau_s
        dS2 = (S1 - S2) / c.tau_s
        Ui = S2 / c.tau_s

        dx1 = -c.ka1 * x1 + c.kb1 * I
        dx2 = -c.ka2 * x2 + c.kb2 * I
        dx3 = -c.ka3 * x3 + c.kb3 * I
        dI = Ui / c.Vi - c.ke * I

        return [dDm1, dDm2, dG, dGt, dG6p, dH, dSRH_S, dS1, dS2, dx1, dx2, dx3, dI]

    def simulate(self, meal_times, meal_durations, meal_cho, bolus_times, bolus_values, bolus_duration=15.0):
        c = HovorkaConstants(BW=self.BW, u_basal=self.u_basal)
        t_span = np.arange(0, c.MAX_TIME, c.h)
        
        # 13-State Initial Arrays
        y0_p = [c.Dm1_0, c.Dm2_0, c.G_0, c.Gt_0, c.G6p_0, c.H_0, c.SRH_S_0, c.S1_0, c.S2_0, c.x1_0, c.x2_0, c.x3_0, c.I_0]
        y0_h = [c.Dm1_0, c.Dm2_0, c.G_0, c.Gt_0, 0.0, 0.0, 0.0, c.S1_0, c.S2_0, c.x1_0, c.x2_0, c.x3_0, c.I_0]

        sol_p = odeint(self.odes, y0_p, t_span, args=(meal_times, meal_durations, meal_cho, bolus_times, bolus_values, bolus_duration, c, "proposed"), rtol=1e-6, atol=1e-8)
        sol_h = odeint(self.odes, y0_h, t_span, args=(meal_times, meal_durations, meal_cho, bolus_times, bolus_values, bolus_duration, c, "hovorka"), rtol=1e-6, atol=1e-8)

        return t_span, sol_p[:, 2], sol_h[:, 2], sol_p[:, 12]

    def plot(self, t, G_proposed, G_hovorka, I):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        fig.patch.set_facecolor('#0e1117')
        for ax in (ax1, ax2):
            ax.set_facecolor('#1a1d27')
            ax.tick_params(colors='#c8ccd4')
            ax.xaxis.label.set_color('#c8ccd4')
            ax.yaxis.label.set_color('#c8ccd4')
            ax.title.set_color('#ffffff')
            for spine in ax.spines.values():
                spine.set_edgecolor('#3a3f4b')

        ax1.plot(t, G_proposed, color='#1f77b4', linewidth=2.0, label='Proposed')
        ax1.plot(t, G_hovorka, color='#d62728', linewidth=1.5, linestyle='--', label='Hovorka')
        ax1.set_title('Blood Glucose Profile Comparison')
        ax1.set_xlabel('Time (min)')
        ax1.set_ylabel('Glucose (mg/dl)')
        ax1.set_xlim(0, 1440)
        ax1.set_ylim(50, 300)
        ax1.legend(facecolor='#1a1d27', edgecolor='#3a3f4b', loc='upper right')
        ax1.grid(True, color='#2a2f3b', linestyle=':', alpha=0.6)

        ax2.plot(t, I, color='#ff7675', linewidth=1.5, label='Plasma Insulin')
        ax2.set_title('Plasma Insulin Profile')
        ax2.set_xlabel('Time (min)')
        ax2.set_ylabel('Insulin (mU/L)')
        ax2.legend(facecolor='#1a1d27', edgecolor='#3a3f4b', loc='upper right')
        ax2.grid(True, color='#2a2f3b', linestyle=':', alpha=0.6)

        plt.tight_layout()
        return fig
