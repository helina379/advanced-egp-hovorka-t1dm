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

        # Explicitly assign both property variants
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
        # BREAKING THE LOOP: Persistence tracking for derivative history step
        self.prev_dGdt = 0.0

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

    def odes(self, y, t, meal_times, meal_durations, meal_cho, bolus_times, bolus_values, bolus_duration):
        Dm1, Dm2, G, Gt, G6p, H, S1, S2, x1, x2, x3, I = y
        c = self.c

        # Force physical bound protection to prevent underflow drops
        G = max(10.0, G)
        G6p = max(0.0, G6p)
        H = max(0.0, H)

        # --- 1. Meal Absorption Subsystem ---
        d_cho = self.meal_input(t, meal_times, meal_durations, meal_cho)
        D_meal = 1000.0 * d_cho / c.Mwg
        
        dDm1 = c.Ag * D_meal - Dm1 / c.tau_d
        dDm2 = Dm1 / c.tau_d - Dm2 / c.tau_d
        Ug = Dm2 / c.tau_d

        # --- 2. Glucose Scaling & Kinetics ---
        Ugc = 18.0 * Ug / c.Vg
        F01uc = 18.0 * c.F01 / c.Vg if G >= 81.0 else (18.0 * c.F01 * G) / (c.Vg * 81.0)
        Erc = c.ke1 * (G - c.Gth) if G >= c.Gth else 0.0
        
        # --- 3. Stable EGP6 Hepatic Subsystem ---
        E = (1.0 - np.tanh((t - c.tD) / c.tau)) / 2.0
        Ggg = (c.Ggg1b + c.Sc * max(0.0, H - c.Hth)) * E
        
        # Using persistent step-state memory history to safely calculate switching boundaries
        if self.prev_dGdt >= 0:
            EGP_val = c.K6gp * G6p - x3 * self.prev_dGdt - c.kp2 * (G - c.Gb)
        else:
            EGP_val = c.K6gp * G6p - c.kp2 * (G - c.Gb)
            
        EGPc = 18.0 * (EGP_val / c.Vg)

        # --- 4. System Differential Expressions ---
        dG = Ugc - F01uc - Erc + (c.k12 * (Gt - c.Gb)) - (x1 * (G - c.Gb)) + EGPc
        dGt = (x1 * (G - c.Gb)) - ((c.k12 + x2) * (Gt - c.Gb))

        dG6p = -c.K6gp * G6p + Ggg + c.Ggng1b
        
        if G >= c.Gb:
            Srhs = c.rho * (0.0 - c.n * c.Hb)
        else:
            Srhb_calc = c.n * c.Hb
            Srhs = c.rho * (0.0 - max(c.sigma * (c.Gth1 - G) / (I + 1.0) + Srhb_calc, 0.0))
            
        Srhd = c.delta * max(-dG, 0.0)
        Srh = Srhs + Srhd
        dH = -c.n * H + Srh

        # Subcutaneous Insulin Absorption
        u = self.insulin_input(t, bolus_times, bolus_values, bolus_duration)
        dS1 = u - S1 * c.k21
        dS2 = (c.k21 * S1) - ((c.kd + c.ka) * S2)
        Ui = S2 / c.tau_s

        # Insulin Remote Action States
        dx1 = -c.ka1 * x1 + c.kb1 * I
        dx2 = -c.ka2 * x2 + c.kb2 * I
        dx3 = -c.ka3 * x3 + c.kb3 * I

        # Circulating Plasma Insulin Compartment
        dI = Ui / c.Vi - c.ke * I

        # Update step memory
        self.prev_dGdt = dG

        return [dDm1, dDm2, dG, dGt, dG6p, dH, dS1, dS2, dx1, dx2, dx3, dI]

    def simulate(self, meal_times, meal_durations, meal_cho, bolus_times, bolus_values, bolus_duration=5.0):
        c = self.c
        t_span = np.arange(0, c.MAX_TIME, c.h)
        self.prev_dGdt = 0.0

        y0 = [
            c.Dm1_0, c.Dm2_0,
            c.G_0, c.Gt_0,
            c.G6p_0, c.H_0,
            c.S1_0, c.S2_0,
            c.x1_0, c.x2_0, c.x3_0,
            c.I_0
        ]

        sol = odeint(
            self.odes, y0, t_span,
            args=(meal_times, meal_durations, meal_cho, bolus_times, bolus_values, bolus_duration),
            rtol=1e-6, atol=1e-8
        )

        G_mgdl = sol[:, 2]
        I = sol[:, 11]
        G_mmol = G_mgdl / 18.0182

        return t_span, G_mmol, G_mgdl, I

    def plot(self, t, G_mmol, G_mgdl, I):
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

        ax1.axhspan(70, 180, alpha=0.12, color='#2ecc71', label='Normal range (70–180 mg/dL)')
        ax1.plot(t, G_mgdl, color='#4fc3f7', linewidth=1.5, label='Plasma Glucose')
        ax1.set_title('Blood Glucose Profile (EGP6 Hovorka Model)')
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('Glucose (mg/dL)')
        ax1.set_ylim(0, 350)  # Bound view to clear clinical window numbers
        ax1.legend(facecolor='#1a1d27', edgecolor='#3a3f4b', loc='upper right')
        ax1.grid(True, color='#2a2f3b', linestyle='--', alpha=0.5)

        ax2.plot(t, I, color='#ff7675', linewidth=1.5, label='Plasma Insulin')
        ax2.set_title('Plasma Insulin Profile')
        ax2.set_xlabel('Time (minutes)')
        ax2.set_ylabel('Insulin (mU/L)')
        ax2.legend(facecolor='#1a1d27', edgecolor='#3a3f4b', loc='upper right')
        ax2.grid(True, color='#2a2f3b', linestyle='--', alpha=0.5)

        plt.tight_layout()
        return fig
