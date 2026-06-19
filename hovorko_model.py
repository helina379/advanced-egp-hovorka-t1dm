import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from hovorka_constants import HovorkaConstants

class HovorkaModel:
    def __init__(self, BW=70.0, u_basal=12.9127):
        self.c = HovorkaConstants(BW=BW, u_basal=u_basal)
        # BUGFIX: Storing u_basal explicitly to handle attribute fallbacks cleanly
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

    def odes(self, y, t, meal_times, meal_durations, meal_cho, bolus_times, bolus_values, bolus_duration):
        """
        State vector y = [Dm1, Dm2, G, Gt, G6p, H, S1, S2, x1, x2, x3, I]
        """
        Dm1, Dm2, G, Gt, G6p, H, S1, S2, x1, x2, x3, I = y
        c = self.c

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

        # Formulating the change in plasma glucose (dG/dt estimation baseline)
        # BUGFIX 1: Tracking plasma glucose derivative dynamics for the EGP switch
        rkg11_est = Ugc - F01uc - Erc + (c.k12 * (Gt - c.Gb)) - (x1 * (G - c.Gb))
        rkg21_est = (x1 * (G - c.Gb)) - ((c.k12 + x2) * (Gt - c.Gb))
        
        # --- 3. Fixed EGP6 Hepatic Subsystem ---
        E = (1.0 - np.tanh((t - c.tD) / c.tau)) / 2.0
        
        # BUGFIX 2: Encapsulated the complete sum inside brackets so baseline Ggg1b decays correctly
        Ggg = (c.Ggg1b + c.Sc * max(0.0, H - c.Hth)) * E
        
        # Evaluating switcher on True Plasma Glucose Derivative (rkg11_est) instead of rkg21_est
        if rkg11_est >= 0:
            EGP_val = c.K6gp * G6p - x3 * rkg11_est - c.kp2 * (G - c.Gb)
        else:
            EGP_val = c.K6gp * G6p - c.kp2 * (G - c.Gb)
            
        EGPc = 18.0 * (EGP_val / c.Vg)

        # --- 4. System Differential Expressions ---
        dG = Ugc - F01uc - Erc + (c.k12 * (Gt - c.Gb)) - (x1 * (G - c.Gb)) + EGPc
        dGt = rkg21_est

        dG6p = -c.K6gp * G6p + Ggg + c.Ggng1b
        
        # Glucagon Regulatory Cascade Equations
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

        return [dDm1, dDm2, dG, dGt, dG6p, dH, dS1, dS2, dx1, dx2, dx3, dI]

    def simulate(self, meal_times, meal_durations, meal_cho, bolus_times, bolus_values, bolus_duration=5.0):
        c = self.c
        t_span = np.arange(0, c.MAX_TIME, c.h)

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