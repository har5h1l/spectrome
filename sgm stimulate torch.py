import numpy as np
import logging
from utils.misc import _set_verbose_level
import torch
import copy

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.hasHandlers():
    ch = logging.StreamHandler() # for console. 
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch) 


param_bounds = {
    "alpha": [0.1, 1], 
    "gei": [0.001, 0.7],
    "gii": [0.001, 2], 
    "speed": [5, 20],
    "taue": [0.005, 0.03], 
    "taug": [0.005, 0.03], 
    "taui": [0.005, 0.20]
}
    
class SGM:
    """
    A SGM model for both FC and PSD forward
    """
    def __init__(self, C, D, freqs, verbose=2, use_logit=False, use_pw=False):
        """args:
            C (array): sc matrix, nroi x nroi
            D (array): dist matrix, nroi x nroi
            freqs: the freqs to get the FC, in Hz. 
            verbose (int, optional): Verbosity level. Defaults to 2.
        """
        _set_verbose_level(verbose, logger)
        if isinstance(freqs, str):
            msg = "only support delta, theta and alpha bands, for other band, plz input a list."
            assert freqs.lower() in ["delta", "theta", "alpha"], msg 
            bands_rgs = {"delta": [2, 3.5], "theta": [4, 7], "alpha": [8, 12]}
            bdlmt = bands_rgs[freqs]
            logger.info(f"The range of {freqs} band is {bdlmt}.")
            freqs = np.linspace(bdlmt[0], bdlmt[1], 10)
        self.C = torch.tensor(C, requires_grad=True)
        self.D = torch.tensor(D, requires_grad=True)
        self.freqs = torch.tensor(freqs, requires_grad=True)

        self.use_logit = use_logit
        self.use_pw=use_pw
        self.L = 0

        '''
        roi = C.shape[0]
        stim_n = len(stim_regions)

        self.stim_arr = np.zeros((roi,1))
        
        for i in range(stim_n):
            self.stim_arr[stim_regions[i]] = 1
        '''

        logger.info(f"Num of ROI is {C.shape[0]}.")
        logger.debug(f"Be careful about your input, freq should be in Hz!")
        logger.debug(f"All tau's params should be in second!")
        
    def _get_lap_result(self, alpha, speed, freq):
        """Get the eig results from Laplaician matrix at give freq
        args:
            alpha (scale): alpha param
            speed (scale): speed param
            freq (float): The freq, in Hz
        """

        C = self.C
        D = self.D
        w = 2 * torch.pi * freq # from Hz to angular
        
        # define sum of degrees for rows and columns for laplacian normalization
        C = C/torch.linalg.norm(C)
        # define sum of degrees for rows and columns for laplacian normalization
        rowdegree = torch.transpose(torch.sum(C, axis=1), 0, -1)
        coldegree = torch.sum(C, axis=0)
    
        degree = (rowdegree + coldegree)/2
        #eps = np.percentile(degree,5)

        #print(f'eps: {eps}')

        eps = 0.01

        nroi = C.shape[0]
        Tau = 0.001 * D / speed
        Cc = C * torch.exp(-1j * Tau * w)
    
        # Eigen Decomposition of Complex Laplacian Here
        L1 = torch.eye(nroi)
        #L2 = torch.divide(1, torch.sqrt(torch.multiply(rowdegree, coldegree)) + eps)
        product = torch.clamp(torch.multiply(rowdegree, coldegree), min=0)  # prevent sqrt of negative
        L2 = torch.divide(1, torch.sqrt(product) + eps)

        L = L1.to(torch.complex128) - alpha * torch.matmul(torch.diag(L2).to(torch.complex128), Cc)
        #L = L1 - alpha * Cc

        d, v = torch.linalg.eig(L) # used to be .eig  
        
        # eig_ind = torch.argsort(torch.abs(d))  # sorting in ascending order and absolute value
        # eig_vec = v[:, eig_ind]  # re-indexing eigen vectors according to sorted index
        # eig_val = d[eig_ind]  # re-indexing eigen values with same sorted index

        eig_vec = v
        eig_val = d
        
        return eig_val, eig_vec
    
    def _get_psd_at_freq(self, params, freq, freq_i, stim_vec):
        """Network Transfer Function for spectral graph model for give freq w (in angular, not in Hz)
           i.e. SGM forward function for a specific w
    
        Args:
            params (dict): parameters for ntf, including alpha, gei, gii, taue, tauG, taui, speed
                              The tau's should in second.
            freq (float): frequency at which to calculate NTF, in Hz
    
        Returns:
            model_out (numpy asarray):  the psd at given freq
    
        """
        params1 = {}
        for ky, v in params.items():
            params1[ky.lower()] = v
        alpha = params1["alpha"]
        gei = params1["gei"]
        gii = params1["gii"]
        taue = params1["taue"]
        tauG = params1["taug"]
        taui = params1["taui"]
        speed = params1["speed"]
        gee = 1
        nroi = self.C.shape[0]
        w = 2 * torch.pi * freq # from Hz to angular
        
        # Defining some other parameters used:
        zero_thr = 0.05
    
        eig_val, eig_vec = self._get_lap_result(alpha, speed, freq)
    
        # Cortical model
        Fe = torch.divide(1 / taue ** 2, (1j * w + 1 / taue) ** 2)
        Fi = torch.divide(1 / taui ** 2, (1j * w + 1 / taui) ** 2)
        FG = torch.divide(1 / tauG ** 2, (1j * w + 1 / tauG) ** 2)
    
        Hed = (1 + (Fe * Fi * gei)/(taue * (1j * w + Fi * gii/taui)))/(1j * w + Fe * gee/taue + (Fe * Fi * gei)**2/(taue * taui * (1j * w + Fi * gii / taui)))
        Hid = (1 - (Fe * Fi * gei)/(taui * (1j * w + Fe * gee/taue)))/(1j * w + Fi * gii/taui + (Fe * Fi * gei)**2/(taue * taui * (1j * w + Fe * gee / taue)))
        Htotal = Hed + Hid
    
        q1 = (1j * w + 1 / tauG * FG * eig_val)
        qthr = zero_thr * torch.abs(q1[:]).max()
        magq1 = torch.maximum(torch.abs(q1), qthr)
        angq1 = torch.angle(q1)
        q1 = torch.multiply(magq1, torch.exp(1j * angq1))
        frequency_response = torch.divide(Htotal, q1)
        
        model_out = 0

        #stim_vec = self.P[:,freq_i].reshape((nroi,1))

        for k in range(nroi):
            model_out += (frequency_response[k]) * torch.outer(eig_vec[:, k], torch.conj(eig_vec[:, k])) #@ stim_vec
            #print(f'In loop: {model_out.shape}')
        #print(f'After loop {model_out.shape}')

        p_noise = 0
        model_power_out = 0

        if not torch.is_tensor(stim_vec):

            stim_vec = torch.from_numpy(stim_vec)

        p_noise = torch.from_numpy(np.eye(nroi))

        if self.use_pw:
            p_noise = stim_vec @ stim_vec.H

        if not torch.is_complex(p_noise):
            p_noise = p_noise.to(torch.complex128)

        #model_out = torch.from_numpy(model_out)

        model_power_out = torch.diag(model_out @ p_noise @ model_out.H)
        #model_power_out = torch.abs(model_power_out) ** 0.5

        eps_stable = 1e-10
        model_power_out = torch.sqrt(torch.abs(model_power_out) + eps_stable)

        # else:

        #     p_noise = np.eye(nroi) + (stim_vec @ np.conjugate(stim_vec).T)

        #     model_power_out = np.diagonal(model_out @ p_noise @ np.conjugate(model_out).T) # @ p_noise
        #     model_power_out = model_power_out ** 0.5

        return model_power_out
    
    def _get_fc_at_freq(self, params, freq, freq_i, stim_vec):
        """Network Transfer Function for spectral graph model.
    
        Args:
            params (dict): params for SGM, including alpha, tauG and speed
            freq (float): frequency at which to calculate FC, in Hz 
    
        Returns:
            fc(numpy asarray):  The FC for the given frequency (freq)
        """
        
        params1 = {}
        for ky, v in params.items():
            params1[ky.lower()] = v
        alpha = params1["alpha"]
        tauG = params1["taug"]
        speed = params1["speed"]
        w = 2*np.pi*freq # change from Hz to angular freq
        nroi = self.C.shape[0]
        
        # Defining some other parameters used:
        zero_thr = 0.05 # in my paper, it is 0.01, but to make it consistent with PSD-SGM, I change it to 0.05
    
        eig_val, eig_vec = self._get_lap_result(alpha, speed, freq)
    
        # Cortical model
        FG = np.divide(1 / tauG ** 2, (1j * w + 1 / tauG) ** 2)
    
    
        q1 = (1j * w + 1 / tauG * FG * eig_val)
        qthr = zero_thr * np.abs(q1[:]).max()
        magq1 = np.maximum(np.abs(q1), qthr)
        angq1 = np.angle(q1)
        q1 = np.multiply(magq1, np.exp(1j * angq1))
        frequency_response = np.divide(1, np.abs(q1)**2)
        
        #fc = eig_vec @ np.diag(frequency_response) @ np.conjugate(eig_vec.T)
        #fc = np.abs(fc)

        U = eig_vec
        U_H = np.conjugate(U.T)

        frequency_response_2 = np.divide(1, q1)

        G = np.diag(frequency_response_2)
        G_H = np.conjugate(G.T)

        #stim_vec = self.P[:,freq_i].reshape((nroi,1))

        PPH = np.eye(nroi) + (stim_vec @ np.conjugate(stim_vec).T)

        fc = U @ G @ U_H @ PPH @ U @ G_H @ U_H

        fc = np.abs(fc)

        return fc
    
    def forward_fc(self, params, stim_array):

        """
        Output:
        estFC, the mean normalized estimated FC at the given frequency computed 
                over the range given in freqrange.
        """
        estFC = 0
        for freq_i, cur_freq in enumerate(self.freqs):
            cur_estFC = self._get_fc_at_freq(params, cur_freq, freq_i, stim_array[:,freq_i].reshape((nroi,1)))
            estFC = cur_estFC/len(self.freqs) + estFC
    
        # Now normalize estFC
        diagFC = np.diag(np.abs(estFC))
        diagFC = 1./np.sqrt(diagFC)
        D = np.diag(diagFC)
        estFC = np.matmul(D, estFC)
        estFC = np.matmul(estFC , np.matrix.getH(D)) # f_ij/\sqrt(f_ii)\sqrt(f_jj)
        estFC = estFC - np.diag(np.diag( estFC ))
    
        return estFC
            
    def forward_psd(self, params, stim_array):
        """run_forward. Function for running the forward model over the passed in range of frequencies,
        for the handed set of parameters (which must be passed in as a dictionary)
    
        Args:
            params (dict): Dictionary of a setting of parameters for the NTF model.
    
        Returns:
            model_out(array): the modelled PSD from SGM model
    
        """
        nroi = self.C.shape[0]

        #params_unscaled = copy.deepcopy(params)

        params_unscaled = {}

        if self.use_logit: # unscale logit transform used in minimization

            #print(params)
            
            for name, v in params.items():
                lower_bnd = param_bounds[name][0]
                upper_bnd = param_bounds[name][1]

                unscaled_val = lower_bnd + ((upper_bnd - lower_bnd) / (1 + torch.exp(-1 * v)))

                params_unscaled[name] = unscaled_val

        else:
            params_unscaled = params

        if not torch.is_tensor(stim_array):

            stim_array = torch.from_numpy(stim_array)

        model_out = torch.zeros(nroi, len(self.freqs))

        for freq_i, freq in enumerate(self.freqs):
            freq_model = self._get_psd_at_freq(params_unscaled, freq, freq_i, stim_array[:,freq_i].reshape((nroi,1)))
            model_out[:,freq_i] = freq_model

        return model_out # removed .real

        # model_out = []
        # for freq_i, freq in enumerate(self.freqs):
        #     freq_model = self._get_psd_at_freq(params, freq, freq_i, stim_array[:,freq_i].reshape((nroi,1)))
        #     model_out.append(freq_model)

        # model_out = np.asarray(model_out).T
        # return model_out.real

    def _get_fft_at_freq(self, params, freq, pw_i):
        """Network Transfer Function for spectral graph model for give freq w (in angular, not in Hz)
           i.e. SGM forward function for a specific w
    
        Args:
            params (dict): parameters for ntf, including alpha, gei, gii, taue, tauG, taui, speed
                              The tau's should in second.
            freq (float): frequency at which to calculate NTF, in Hz
    
        Returns:
            model_out (numpy asarray):  the psd at given freq
    
        """
        params1 = {}
        for ky, v in params.items():
            params1[ky.lower()] = v
        alpha = params1["alpha"]
        gei = params1["gei"]
        gii = params1["gii"]
        taue = params1["taue"]
        tauG = params1["taug"]
        taui = params1["taui"]
        speed = params1["speed"]
        gee = 1
        nroi = self.C.shape[0]
        w = 2 * torch.pi * freq # from Hz to angular
        
        # Defining some other parameters used:
        zero_thr = 0.05
    
        eig_val, eig_vec = self._get_lap_result(alpha, speed, freq)
    
        # Cortical model
        Fe = torch.divide(1 / taue ** 2, (1j * w + 1 / taue) ** 2)
        Fi = torch.divide(1 / taui ** 2, (1j * w + 1 / taui) ** 2)
        FG = torch.divide(1 / tauG ** 2, (1j * w + 1 / tauG) ** 2)
    
        Hed = (1 + (Fe * Fi * gei)/(taue * (1j * w + Fi * gii/taui)))/(1j * w + Fe * gee/taue + (Fe * Fi * gei)**2/(taue * taui * (1j * w + Fi * gii / taui)))
        Hid = (1 - (Fe * Fi * gei)/(taui * (1j * w + Fe * gee/taue)))/(1j * w + Fi * gii/taui + (Fe * Fi * gei)**2/(taue * taui * (1j * w + Fe * gee / taue)))
        Htotal = Hed + Hid
    
        q1 = (1j * w + 1 / tauG * FG * eig_val)
        qthr = zero_thr * torch.abs(q1[:]).max()
        magq1 = torch.maximum(torch.abs(q1), qthr)
        angq1 = torch.angle(q1)
        q1 = torch.multiply(magq1, torch.exp(1j * angq1))
        frequency_response = torch.divide(Htotal, q1)
        
        model_out = 0

        #stim_vec = self.P[:,freq_i].reshape((nroi,1))

        if not torch.is_tensor(pw_i):
            pw_i = torch.from_numpy(pw_i)

        for k in range(nroi):
            model_out += (frequency_response[k]) * torch.outer(eig_vec[:, k], torch.conj(eig_vec[:, k])) @ pw_i

        #print(model_out.shape)

        return model_out[:,0] # to get tensor shape right

    def forward_ft(self, params, stim_array, freqs_ft): # note, stim_array needs to match dim of freqs_ft

        nroi = self.C.shape[0]

        params_unscaled = {}

        if self.use_logit: # unscale logit transform used in minimization
            
            for name, v in params.items():
                lower_bnd = param_bounds[name][0]
                upper_bnd = param_bounds[name][1]

                unscaled_val = lower_bnd + ((upper_bnd - lower_bnd) / (1 + torch.exp(-1 * v)))

                params_unscaled[name] = unscaled_val

        else:
            params_unscaled = params

        if not torch.is_tensor(stim_array):

            stim_array = torch.from_numpy(stim_array)

        ft_out = torch.zeros(nroi, len(freqs_ft), dtype=complex)

        for freq_i, freq in enumerate(freqs_ft):
            freq_model = self._get_fft_at_freq(params_unscaled, freq, stim_array[:,freq_i].reshape((nroi,1)))
            ft_out[:,freq_i] = freq_model

        return ft_out

