class HovorkaConstants:
    def __init__(self, BW=70.0, u_basal=12.9127):
        # Patient parameters
        self.BW = BW                        # Body weight (kg)

        # Glucose subsystem
        self.F01 = 0.00097 * BW             # Insulin-independent glucose flux
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

        # Initial conditions
        self.S1_0 = u_basal * self.tau_s    
        self.S2_0 = u_basal * self.tau_s    
        self.I_0 = u_basal / (0.01656 * BW) 
        self.x1_0 = 0.30898 * u_basal / BW  
        self.x2_0 = 0.04951 * u_basal / BW  
        self.x3_0 = 3.2206 * u_basal / BW   
        self.G_0 = 90.0                     # Initial Plasma Glucose G (mg/dL)
        self.Gt_0 = 90.0                    # Initial Tissue Glucose Gt (mg/dL)
        self.Dm1_0 = 0.0
        self.Dm2_0 = 0.0
        self.G6p_0 = 41.897                 # Initial Glucose-6-Phosphate state
        self.H_0 = 58.0e-7                  # Initial Glucagon state

        # Simulation Configuration
        self.MAX_TIME = 1440              
        self.h = 0.1