import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class HovorkaConstants:
    def __init__(self, BW=70.0, u_basal=12.9127):
        self.BW = BW                        # Body weight (kg)

        # --- Glucose Subsystem (Table 2) ---
        self.F01 = 0.00097 * BW             # Insulin-independent glucose flux (F_ii)
        self.Vg = 0.16 * BW                 # Distribution volume of blood glucose
        self.k12 = 0.066                    # Transfer rate Q2->Q1
        self.Mwg = 180.0                    # Molecular weight of glucose (M_wt)
        self.Gb = 90.0                      # Basal Glucose (G(0))
        self.Gth = 162.0                    # Renal threshold
        self.Gth1 = 60.0                    # Hypoglycemic threshold (G_th1)
        self.ke1 = 0.007                    # Glomerular filtration rate (K_e1)

        # --- Hepatic EGP Parameters (Table 3) ---
        self.EGP_b = 1.23                   # Basal value of EGP
        self.K6gp = 0.034                   # Rate of dephosphorylation (KG6p)
        self.Ggg1b = 0.7425                 # Basal EGP glycogenolysis contribution
        self.Ggng1b = 0.495                 # Basal EGP glyconeogenesis contribution (G_GNG,b)
        self.Sc = 297.0                     # Sensitivity of glycogenolysis to glucagon
        self.Hth = 80.0e-7                  # Plasma glucagon threshold value
        self.tD = 59.90                     # Time of onset of evanescence
        self.tau = 23.24                    # Time constant of evanescence effect (T)
        self.g6po = 5.50                    # Rapid increase parameter
        self.kp2 = 0.0007                   # Liver glucose effectiveness
        self.n = 0.01                       # Glucagon clearance rate
        self.rho = 0.86                     # Glucagon secretion relaxation rate (rho)
        self.Hb = 58.0e-7                   # Basal glucagon
        self.sigma = 1.714410e-11           # Glucagon stimulation by hypoglycemia
        self.delta = 0.98e-7                # Dynamic (rate-of-fall) glucagon stimulation

        # --- Insulin Subsystem Parameters (Table 4) ---
        self.Vi = 0.12 * BW                 # Distribution volume of insulin
        self.tau_s = 55.0                   # Time constant for insulin absorption
        self.ke = 0.138                     # Fractional elimination rate of insulin
        self.k21 = 0.045                    # Inter-compartmental transfer rate
        self.kd = 0.0021
        self.ka = 0.02

        # --- Meal Absorption Parameters (Table 2) ---
        self.Ag = 0.8                       # Carbohydrate utilization factor
        self.tau_d = 40.0                   # Meal absorption time constant (TD)

        # --- Insulin Action Parameters (Table 4) ---
        self.ka1 = 0.006                    # Deactivation rate x1
        self.ka2 = 0.06                     # Deactivation rate x2
        self.ka3 = 0.03                     # Deactivation rate x3
        self.kb1 = 3.072e-5                 # Activation rate constant
        self.kb2 = 4.92e-5                  # Activation rate constant
        self.kb3 = 0.00156                  # Activation rate constant

        self.u_basal = u_basal
        self.u0 = u_basal

        # --- VERIFIED INITIAL CONDITIONS (Page 12 Formulas) ---
        self.S1_0 = u_basal * self.tau_s    # s1(0) = tau_s * u(0)
        self.S2_0 = u_basal * self.tau_s    # s2(0) = tau_s * u(0)
        self.I_0 = u_basal / (0.01656 * BW) # I(0) = u(0) / (0.01656 * BW)
        self.x1_0 = 0.30898 * u_basal / BW  # x1(0) = 0.30898 * u(0) / BW
        self.x2_0 = 0.04951 * u_basal / BW  # x2(0) = 0.04951 * u(0) / BW
        self.x3_0 = 3.2206 * u_basal / BW   # x3(0) = 3.2206 * u(0) / BW

        self.G_0 = 90.0                     # G(0) = 90 mg/dL
        self.Gt_0 = 70.0                    # G1(0) = 70 mg/dL (Table 2)
        self.Dm1_0 = 0.0
        self.Dm2_0 = 0.0
        self.G6p_0 = (self.EGP_b / self.K6gp) + self.g6po  # G6P(0), Eq. 13
        self.H_0 = 58.0e-7                  # Basal glucagon state (Hb)

        # SRb_H = n*Hb (per "SRS_Hb = SRb_H = nHb", glucagon section)
        # SRs(0) initialized at its basal steady-state value
        self.SRb_H = self.n * self.Hb
        self.Srs_0 = self.SRb_H

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
                d_cho = meal_cho[i] / meal_durations[i]
                break
        return d_cho

    def insulin_input(self, t, bolus_times, bolus_values, bolus_duration=15.0):
        for i in range(len(bolus_times)):
            if bolus_times[i] <= t < bolus_times[i] + bolus_duration:
                return bolus_values[i]
        return self.u_basal

    def odes(self, y, t, meal_times, meal_durations, meal_cho, bolus_times, bolus_values, bolus_duration, model_type="proposed"):
        Dm1, Dm2, G, Gt, G6p, H, Srs, S1, S2, x1, x2, x3, I = y
        c = self.c

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

        B = Ugc - F01uc - Erc + (c.k12 * (Gt - c.Gb)) - (x1 * (G - c.Gb))

        # --- 3. Hepatic Glucose Production (EGP) ---
        # NOTE: EGP(t) is already in mg/dl/min per Table 3 / Eq. 5 - it must NOT be
        # scaled by 18/Vg (that conversion is only for the mmol-based UG/F01/ER terms,
        # which is why Erc above already skips it). x3 is unit-less and acts directly
        # on dG/dt, not on a molar flux.
        if model_type == "proposed":
            E = (1.0 - np.tanh((t - c.tD) / c.tau)) / 2.0
            Ggg = (c.Ggg1b + c.Sc * max(0.0, H - c.Hth)) * E
            dG6p = -c.K6gp * G6p + Ggg + c.Ggng1b

            EGP_base = c.K6gp * G6p - c.kp2 * (G - c.Gb)

            # Implicit solve: dG/dt = EGP(t) + B, with EGP(t) = EGP_base - x3*dG/dt (if dG/dt>=0)
            # => EGP(t) = (EGP_base - x3*B) / (1+x3); dG/dt = (EGP_base + B) / (1+x3)
            dGdt_trial = (EGP_base + B) / (1.0 + x3)
            if dGdt_trial >= 0:
                EGP_val = (EGP_base - x3 * B) / (1.0 + x3)
            else:
                EGP_val = EGP_base
        else:
            # Classic Hovorka comparison baseline: EGP(t) = EGP0 - x3*EGP0
            EGP_val = c.EGP_b * (1.0 - x3)
            dG6p = 0.0

        # --- 4. Differential Equations ---
        dG = B + EGP_val
        dGt = (x1 * (G - c.Gb)) - ((c.k12 + x2) * (Gt - c.Gb))

        if model_type == "proposed":
            # SRb_H = n*Hb is the basal target that SRs(t) relaxes toward
            if G >= c.Gb:
                SRs_target = c.SRb_H
            else:
                SRs_target = max(c.sigma * (c.Gth1 - G) / (I + 1.0) + c.SRb_H, 0.0)

            # SRs(t) is a dynamic (lagged) state, NOT an instantaneous value
            dSrs = -c.rho * (Srs - SRs_target)

            # SRd(t): dynamic stimulation from rate of glucose fall (instantaneous, uses dG computed above)
            Srd = c.delta * max(-dG, 0.0)

            dH = -c.n * H + (Srs + Srd)
        else:
            dSrs = 0.0
            dH = 0.0

        # Subcutaneous Insulin Channels
        u = self.insulin_input(t, bolus_times, bolus_values, bolus_duration)
        dS1 = u - S1 / c.tau_s
        dS2 = (S1 - S2) / c.tau_s
        Ui = S2 / c.tau_s

        # Actions & Plasma Clearances
        dx1 = -c.ka1 * x1 + c.kb1 * I
        dx2 = -c.ka2 * x2 + c.kb2 * I
        dx3 = -c.ka3 * x3 + c.kb3 * I
        dI = Ui / c.Vi - c.ke * I

        return [dDm1, dDm2, dG, dGt, dG6p, dH, dSrs, dS1, dS2, dx1, dx2, dx3, dI]

    def simulate(self, meal_times, meal_durations, meal_cho, bolus_times, bolus_values, bolus_duration=15.0):
        c = self.c
        t_span = np.arange(0, c.MAX_TIME, c.h)

        # --- Run 1: Proposed Model Simulation ---
        y0_p = [c.Dm1_0, c.Dm2_0, c.G_0, c.Gt_0, c.G6p_0, c.H_0, c.Srs_0,
                c.S1_0, c.S2_0, c.x1_0, c.x2_0, c.x3_0, c.I_0]
        # hmax is critical: without it, LSODA can take internal steps larger than the
        # 5-20 min meal/bolus pulse windows and step clean over them, silently zeroing
        # out the forcing inputs even though the output grid looks smooth.
        sol_p = odeint(self.odes, y0_p, t_span,
                        args=(meal_times, meal_durations, meal_cho, bolus_times, bolus_values, bolus_duration, "proposed"),
                        rtol=1e-6, atol=1e-8, hmax=1.0)
        G_proposed = sol_p[:, 2]
        I_proposed = sol_p[:, 12]

        # --- Run 2: Classic Hovorka Simulation ---
        y0_h = [c.Dm1_0, c.Dm2_0, c.G_0, c.Gt_0, 0.0, 0.0, 0.0,
                c.S1_0, c.S2_0, c.x1_0, c.x2_0, c.x3_0, c.I_0]
        sol_h = odeint(self.odes, y0_h, t_span,
                        args=(meal_times, meal_durations, meal_cho, bolus_times, bolus_values, bolus_duration, "hovorka"),
                        rtol=1e-6, atol=1e-8, hmax=1.0)
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

        # --- PLOT ONLY PROPOSED ---
        ax1.plot(t, G_proposed, color='#1f77b4', linewidth=2.0, label='Proposed')
        
        ax1.set_title('Blood Glucose Profile')
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
