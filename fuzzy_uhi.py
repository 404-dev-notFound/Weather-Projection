import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class UHIFuzzyAdjuster:
    def __init__(self):
        """
        Initializes the Mamdani Fuzzy Inference System for Urban Heat Island adjustments.
        Using the scikit-fuzzy library.
        """
        # 1. Define Antecedents (Inputs)
        # T_base: Base Temperature from CNN output (Celsius). Range 10-55.
        self.t_base = ctrl.Antecedent(np.arange(10, 56, 1), 't_base')
        
        # D_urban: Distance from Urban Core using grid cells. Extends outwards up to 24 units.
        self.d_urban = ctrl.Antecedent(np.arange(0, 25, 1), 'd_urban')
        
        # 2. Define Consequent (Output)
        # UHI_Adjustment: Degrees Celsius to add (+0 to +3)
        self.uhi_adj = ctrl.Consequent(np.arange(0, 4, 0.1), 'uhi_adj', defuzzify_method='centroid')

        # 3. Fuzzy Sets Configuration
        # Temperature Logic: UHI effects are often worse during already hot background conditions.
        self.t_base['Mild'] = fuzz.trapmf(self.t_base.universe, [10, 10, 25, 30])
        self.t_base['Warm'] = fuzz.trimf(self.t_base.universe, [25, 35, 45])
        self.t_base['Hot'] = fuzz.trapmf(self.t_base.universe, [35, 45, 55, 55])
        
        # Distance Logic: Core vs Suburbs vs Deep Desert
        self.d_urban['Core'] = fuzz.trapmf(self.d_urban.universe, [0, 0, 2, 5])
        self.d_urban['Suburb'] = fuzz.trimf(self.d_urban.universe, [2, 6, 12])
        self.d_urban['Desert'] = fuzz.trapmf(self.d_urban.universe, [8, 14, 25, 25])
        
        # UHI Adjustment Degrees
        self.uhi_adj['Zero'] = fuzz.trimf(self.uhi_adj.universe, [0, 0, 0.5])
        self.uhi_adj['Moderate'] = fuzz.trimf(self.uhi_adj.universe, [0.2, 1.5, 2.5])
        self.uhi_adj['Severe'] = fuzz.trapmf(self.uhi_adj.universe, [2.0, 3.0, 4.0, 4.0])

        # 4. Define Rule Base
        rule1 = ctrl.Rule(self.d_urban['Core'] & self.t_base['Hot'], self.uhi_adj['Severe'])
        rule2 = ctrl.Rule(self.d_urban['Core'] & self.t_base['Warm'], self.uhi_adj['Moderate'])
        rule3 = ctrl.Rule(self.d_urban['Suburb'] & self.t_base['Hot'], self.uhi_adj['Moderate'])
        rule4 = ctrl.Rule(self.d_urban['Desert'], self.uhi_adj['Zero'])
        rule5 = ctrl.Rule(self.t_base['Mild'], self.uhi_adj['Zero'])

        self.uhi_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
        self.uhi_sim = ctrl.ControlSystemSimulation(self.uhi_ctrl)

    def compute_distance_matrix(self, grid_size=17, core_idx=(12, 10)):
        """
        Compute Euclidean distance of each pixel from the Dubai Urban Core.
        Assuming Core is around index [12, 10] relative to UAE bounding box, 
        but this can be parameterized.
        """
        dist_matrix = np.zeros((grid_size, grid_size))
        for i in range(grid_size):
            for j in range(grid_size):
                dist_matrix[i, j] = np.sqrt((i - core_idx[0])**2 + (j - core_idx[1])**2)
        return dist_matrix

    def adjust_temperature_grid(self, temp_grid, core_idx=(12, 10)):
        """
        Applies the UHI fuzzy adjustment to a single 17x17 temperature grid mathematically,
        blending the core heat into the desert seamlessly.
        """
        adjusted_grid = np.copy(temp_grid)
        grid_size = temp_grid.shape[0]
        dist_matrix = self.compute_distance_matrix(grid_size, core_idx)
        
        for i in range(grid_size):
            for j in range(grid_size):
                t = temp_grid[i, j]
                d = dist_matrix[i, j]
                
                # Clip input to universe specs
                t = np.clip(t, 10, 55)
                d = np.clip(d, 0, 24)
                
                self.uhi_sim.input['t_base'] = t
                self.uhi_sim.input['d_urban'] = d
                
                # Compute fuzzy centroid
                try:
                    self.uhi_sim.compute()
                    adj = self.uhi_sim.output['uhi_adj']
                except Exception as e:
                    adj = 0 # Fallback in case fuzzy sim hits gaps
                    
                adjusted_grid[i, j] += adj
                
        return adjusted_grid
