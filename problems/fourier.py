from cmath import phase
import torch
import torch.fft
import numpy as np
from problems.problem import ForwardOperator

class FourierOperator(ForwardOperator):
    def __init__(self, hparams):
        super().__init__(hparams)
    
    def _make_noise(self):
        if self.hparams.problem.add_noise and self.hparams.problem.noise_type == 'gaussian_nonwhite':
            noise_vars = torch.rand(self.hparams.problem.num_measurements * 2) #times 2 to account for real/im
        else:
            noise_vars = None
        
        return noise_vars

    def add_noise(self, Ax):
        """
        Constructs and returns additive noise if the problem calls for it.

        Args:
            Ax: The measurements to add noise to.
                Torch tensor [N, self.hparams.problem.y_shape].
        """
        if self.noisy:
            if self.hparams.problem.noise_type == 'gaussian':
                noise = torch.randn(self.hparams.problem.num_measurements * 2).type_as(Ax) * self.hparams.problem.noise_std
            elif self.hparams.problem.noise_type == 'gaussian_nonwhite': 
                noise = torch.randn(self.hparams.problem.num_measurements * 2).type_as(Ax) * self.hparams.problem.noise_std * self.noise_vars.type_as(Ax)
            else:
                raise NotImplementedError('unsupported type of additive noise')
        
            return Ax + noise.view(Ax.shape[1:])
        else:
            return Ax
    
    def fft(self, x):
        """Performs a centered and orthogonal fft in torch >= 1.7"""
        x = torch.fft.fftshift(x, dim=(-2, -1))
        x = torch.fft.fft2(x, dim=(-2, -1), norm='ortho')
        x = torch.fft.ifftshift(x, dim=(-2, -1))
        return x
        
    def ifft(self, x):
        """Performs a centered and orthogonal ifft in torch >= 1.7"""
        x = torch.fft.ifftshift(x, dim=(-2, -1))
        x = torch.fft.ifft2(x, dim=(-2, -1), norm='ortho')
        x = torch.fft.fftshift(x, dim=(-2, -1))
        return x
    
    def _make_fft_mask(self):
        image_size = self.hparams.data.image_size
        m = self.hparams.problem.num_measurements // self.hparams.data.num_channels #divide by 3 since we multiply by 3 when reading config
    
        if self.hparams.problem.fourier_mask_type == 'radial':
            raise NotImplementedError('Radial mask orientation not supported')
        elif self.hparams.problem.fourier_mask_type == 'horizontal':
            raise NotImplementedError('Horizontal mask orientation not supported')
        elif self.hparams.problem.fourier_mask_type == 'vertical':
            raise NotImplementedError('Vertical mask orientation not supported')
        elif self.hparams.problem.fourier_mask_type == 'random':
            mask = torch.zeros(image_size**2)
            nonzero_idx = np.random.choice(image_size**2, m, replace=False)
            mask[nonzero_idx] = 1
            mask = mask.view(image_size, image_size)
        else:
            raise NotImplementedError('Fourier mask orientation not supported')
        
        return mask
    
    def _make_A(self):
        A_linear = None #TODO replace this with the subsampled orthonormal FFT matrix
        A_mask = self._make_fft_mask()
        
        def A_functional(x):
            #TODO update this
            out = self.A_mask * self.fft(x) #[N, C, H, W]

            return out.flatten(start_dim=1)[:, self.kept_inds] #[N, m]

        A_dict = {'linear': A_linear,
                  'mask': A_mask,
                  'functional': A_functional}
        
        return A_dict

    def forward(self, x, targets=False):
        Ax = self.A_mask * self.fft(x) #[N, C, H, W] torch.complex64
        Ax = torch.view_as_real(Ax) #[N, C, H, W, 2] torch float32

        #If we are performing sample selection, then we would like to return [N, C, H, W, 2]
        #Else return a flattened version
        if not self.hparams.problem.learn_samples:
            Ax = Ax.flatten(start_dim=2, end_dim=-2) #[N, C, HW, 2]
            Ax = Ax[:, :, self.kept_inds, :] #[N, C, m // C, 2] 
            Ax = Ax.flatten(start_dim=1) #[N, 2m]

        if targets:
            Ax = self.add_noise(Ax)
        
        return Ax

    def adjoint(self, vec):
        #TODO make this more efficient (if needed)
        out_c, out_h, out_w = self.hparams.image_shape
        ans = torch.mm(self.A_linear.T, vec.T).T #[N, n] #NOTE we have to define A_linear first!

        return ans.view(-1, out_c, out_h, out_w) #[N, C, H, W]

    @torch.no_grad()
    def get_measurements_image(self, x, targets=False, c=None):
        """Returns the magnitude and phase fft image as well as reconstruction from subsampled fft coeffs"""
        orig_shape = list(x.shape) #[N, C, H, W]
        orig_shape.append(2) #to account for complex operations

        if c is None:
            Ax = self.A_mask * self.fft(x) #[N, C, H, W] torch.complex64
        else:
            Ax = c * self.fft(x) #[N, C, H, W] torch.complex64
        Ax = torch.view_as_real(Ax) #[N, C, H, W, 2] torch float32
        Ax = Ax.flatten(start_dim=2, end_dim=-2) #[N, C, HW, 2]

        if targets:
            Ax[:, :, self.kept_inds, :] = self.add_noise(Ax[:, :, self.kept_inds, :])
        
        Ax = Ax.view(orig_shape) #[N, C, H, W, 2]
        Ax = torch.view_as_complex(Ax) #[N, C, H, W]
        
        mag_img = torch.abs(Ax)
        phase_img = torch.angle(Ax)
        inverted_img = torch.abs(self.ifft(Ax))

        out_dict = {"Mag": mag_img,
                    "Phase": phase_img,
                    "Inverted": inverted_img}
        
        return out_dict
