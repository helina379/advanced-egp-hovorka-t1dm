import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class HovorkaConstants:
    def __init__(self, BW=70.0, u_basal=12.9127):
        self.BW = BW                        # Body weight (kg)

        # Glucose subsystem
        self.F01 = 0.00097 * BW             # Non-insulin dependent glucose flux
        self.Vg = 0.16 * BW                 # Distribution volume of glucose (L)
        self.k12 = 0.066                    # Transfer rate Q2->Q1 (min^-1)
        self.Mwg = 180.0                    # Molecular weight of glucose (g/mol)
        self.Gb = 90.0                      # Basal Glucose (mg/dL)
        self.Gth = 162.0                    # Renal threshold (mg/dL)
        self.Gth1 = 60.0                    # Hypoglycemic threshold (mg/dL)
        
        # Rates from EGP model spec
        self.kp2 = 0.0007                   # Liver glucose effectiveness
        self.ke1 = 0.007                    # Glomerular filtration rate
        self.EGP_b = 1.23                   # Basal EGP for classic Hovorka comparison
        
        # Insulin subsystem
        self.Vi = 0.12 * BW                 # Distribution volume of insulin (L)
        self.tau_s = 55.0                   # Time constant for SC insulin absorption (min)
        self.ke = 0.138                     # Insulin elimination rate (min^-1)
        self.k21 = 0.045                    # Inter-compartmental insulin transfer rate
        self.kd = 0.0021                    
        self.ka = 0.02                      

        # Meal absorption
        self.Ag = 0.8                       # CHO bioavailability
        self.tau_d = 40.0                   # Time constant for meal absorption (min)

        # Insulin action
        self.ka1 = 0.006                    # Deactivation rate x1 (min^-1)
        self.ka2 = 0.06                     # Deactivation rate x2 (min^-1)
        self.ka3 = 0.03                     # Deactivation rate x3 (min^-1)
        self.kb1 = 3.072e-5                 # Activation Rate constants
        self.kb2 = 4.92e-5                  
        self.kb3 = 0.00156                  

        # EGP Hepatic Subsystem Parameters
        self.K6gp = 0.034                   
        self.Sc = 297.0                     
        self.Hth = 80.0e-7                  # Glucagon threshold value
        self.tD = 59.90                     # Time of onset of evanescence
        self.tau = 23.24                    # Time constant of evanescence
        self.Ggng1b = 0.495                 # Basal glyconeogenesis
        self.Ggg1b = 0.7425                 # Basal glycogenolysis
        self.n = 0.01                       # Glucagon clearance rate
        self.rho = 0.86                     
        self.sigma = 1.714410e-11           # Sigma scale parameter
        self.delta = 0.98e-7                # Delta parameter
        self.Hb = 58.0e-7                   # Basal glucagon

        self.u_basal = u_basal            
        self.u0 = u_basal

        # Initial conditions
        self.S1_0 = u_basal * self.tau_s    
        self.S2_0 = u_basal * self.tau_s    
        self.I_0 = u_basal / (0.01656 * BW) 
        self.x1_0 = 0.30898 * u_basal / BW  
        self.x2_0 = 0.04951 * u_basal / BW  
        self.x3_0 = 3.2206 * u_basal / BW   
        self.G_0 = 90.0                     
        self.Gt_0 = 90.0                    
        self.Dm1_0 = 0.0
        self.Dm2_0 = 0.0
        self.G6p_0 = 41.897                 
        self.H_0 = 58.0e-7                  

        # Simulation Configuration
        self.MAX_TIME = 1440              
        self.h = 0.1                        

class HovorkaModel:
    def __init__(self, BW=70.0, u_basal=12.9127):
        self.c = HovorkaConstants(BW=BW, u_basal=u_basal)
        self.u_basal = u_basal

    def meal_input(self, t, meal_times, meal_durations, meal_cho):
        d_cho = 0.0
        for i in range(len(meal_times)):
            if meal_times[i] <= t < meal_times[i] + meal_durations[i]:
                d_cho = meal_cho[i]
                break
        return d_cho

    def insulin_input(self, t, bolus_times, bolus_values, bolus_duration=5.0):
        for i in range(len(bolus_times)):
            if bolus_times[i] <= t < bolus_times[i] + bolus_duration:
                return bolus_values[i]
        return self.u_basal

    def odes(self, y, t, meal_times, meal_durations, meal_cho, bolus_times, bolus_values, bolus_duration, model_type="proposed"):
        Dm1, Dm2, G, Gt, G6p, H, S1, S2, x1, x2, x3, I = y
        c = self.c

        G = max(10.0, G)
        G6p = max(0.0, G6p)
        H = max(0.0, H)

        # --- 1. Meal Absorption ---
        d_cho = self.meal_input(t, meal_times, meal_durations, meal_cho)
        D_meal = 1000.0 * d_cho / c.Mwg
        
        dDm1 = c.Ag * D_meal - Dm1 / c.tau_d
        dDm2 = Dm1 / c.tau_d - Dm2 / c.tau_d
        Ug = Dm2 / c.tau_d

        # --- 2. Glucose Scaling & Kinetics ---
        Ugc = 18.0 * Ug / c.Vg
        F01uc = 18.0 * c.F01 / c.Vg if G >= 81.0 else (18.0 * c.F01 * G) / (c.Vg * 81.0)
        Erc = c.ke1 * (G - c.Gth) if G >= c.Gth else 0.0
        
        # Kinetic base movement (B)
        B = Ugc - F01uc - Erc + (c.k12 * (Gt - c.Gb)) - (x1 * (G - c.Gb))
        
        # --- 3. Hepatic EGP Choice ---
        if model_type == "proposed":
            E = (1.0 - np.tanh((t - c.tD) / c.tau)) / 2.0
            Ggg = (c.Ggg1b + c.Sc * max(0.0, H - c.Hth)) * E
            dG6p = -c.K6gp * G6p + Ggg + c.Ggng1b
            
            EGP_base = c.K6gp * G6p - c.kp2 * (G - c.Gb)
            
            # Pure algebraic lookahead breakdown to avoid feedback loop instability
            dGdt_positive_regime = (B + 18.0 * EGP_base / c.Vg) / (1.0 + 18.0 * x3 / c.Vg)
            
            if dGdt_positive_regime >= 0:
                EGP_val = EGP_base - x3 * dGdt_positive_regime
            else:
                EGP_val = EGP_base
                
            EGPc = 18.0 * (EGP_val / c.Vg)
        else:
            # Classic Hovorka baseline model
            EGP_val = c.EGP_b * np.exp(-x3)
            EGPc = 18.0 * EGP_val / c.Vg
            dG6p = 0.0

        # --- 4. Differential Expressions ---
        dG = B + EGPc
        dGt = (x1 * (G - c.Gb)) - ((c.k12 + x2) * (Gt - c.Gb))
        
        if model_type == "proposed":
            if G >= c.Gb:
                Srhs = c.rho * (0.0 - c.n * c.Hb)
            else:
                Srhb_calc = c.n * c.Hb
                Srhs = c.rho * (0.0 - max(c.sigma * (c.Gth1 - G) / (I + 1.0) + Srhb_calc, 0.0))
            Srhd = c.delta * max(-dG, 0.0)
            dH = -c.n * H + (Srhs + Srhd)
        else:
            dH = 0.0

        # Subcutaneous Channel
        u = self.insulin_input(t, bolus_times, bolus_values, bolus_duration)
        dS1 = u - S1 * c.k21
        dS2 = (c.k21 * S1) - ((c.kd + c.ka) * S2)
        Ui = S2 / c.tau_s

        # Actions & Circulating Plasma Insulin
        dx1 = -c.ka1 * x1 + c.kb1 * I
        dx2 = -c.ka2 * x2 + c.kb2 * I
        dx3 = -c.ka3 * x3 + c.kb3 * I
        dI = Ui / c.Vi - c.ke * I

        return [dDm1, dDm2, dG, dGt, dG6p, dH, dS1, dS2, dx1, dx2, dx3, dI]

    def simulate(self, meal_times, meal_durations, meal_cho, bolus_times, bolus_values, bolus_duration=5.0):
        c = self.c
        t_span = np.arange(0, c.MAX_TIME, c.h)
        
        # --- Run 1: Proposed Model Simulation ---
        y0_p = [c.Dm1_0, c.Dm2_0, c.G_0, c.Gt_0, c.G6p_0, c.H_0, c.S1_0, c.S2_0, c.x1_0, c.x2_0, c.x3_0, c.I_0]
        sol_p = odeint(self.odes, y0_p, t_span, args=(meal_times, meal_durations, meal_cho, bolus_times, bolus_values, bolus_duration, "proposed"), rtol=1e-6, atol=1e-8)
        G_proposed = sol_p[:, 2]
        I_proposed = sol_p[:, 11]

        # --- Run 2: Classic Hovorka Simulation ---
        y0_h = [c.Dm1_0, c.Dm2_0, c.G_0, c.Gt_0, 0.0, 0.0, c.S1_0, c.S2_0, c.x1_0, c.x2_0, c.x3_0, c.I_0]
        sol_h = odeint(self.odes, y0_h, t_span, args=(meal_times, meal_durations, meal_cho, bolus_times, bolus_values, bolus_duration, "hovorka"), rtol=1e-6, atol=1e-8)
        G_hovorka = sol_h[:, 2]

        return t_span, G_proposed, G_hovorka, I_proposed

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

        # Top Plot: Overlaid curves matching Mam's template exactly
        ax1.plot(t, G_proposed, color='#1f77b4', linewidth=2.0, label='Proposed')
        ax1.plot(t, G_hovorka, color='#d62728', linewidth=1.5, linestyle='--', label='Hovorka')
        ax1.set_title('Blood Glucose Profile Comparison')
        ax1.set_xlabel('Time (min)')
        ax1.set_ylabel('Glucose (mg/dl)')
        ax1.set_xlim(0, 1440)
        ax1.set_ylim(50, 300)
        ax1.legend(facecolor='#1a1d27', edgecolor='#3a3f4b', loc='upper right')
        ax1.grid(True, color='#2a2f3b', linestyle=':', alpha=0.6)

        # Bottom Plot: Plasma Insulin Actions
        ax2.plot(t, I, color='#ff7675', linewidth=1.5, label='Plasma Insulin')
        ax2.set_title('Plasma Insulin Profile')
        ax2.set_xlabel('Time (min)')
        ax2.set_ylabel('Insulin (mU/L)')
        ax2.legend(facecolor='#1a1d27', edgecolor='#3a3f4b', loc='upper right')
        ax2.grid(True, color='#2a2f3b', linestyle=':', alpha=0.6)

        plt.tight_layout()
        return fig
