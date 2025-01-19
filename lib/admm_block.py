import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import math
from lib.backup_modules import LR_guess, k_hop_neighbors

class ADMMBlock(nn.Module):

    def __init__(self, T, n_nodes, n_heads, n_channels, nearest_nodes, device,
                 ADMM_info = {
                 'ADMM_iters':50,
                 'CG_iters': 3,
                 'PGD_iters': 3,
                 'mu_u_init':10,
                 'mu_d1_init':10,
                 'mu_d2_init':10,
                 }, 
                 ablation='None'):
        super().__init__()
        # edges and edge_weights constructed as self variables
        self.device = device
        self.T = T
        self.n_nodes = n_nodes
        self.n_heads = n_heads
        self.n_channels = n_channels
        # graphs (edges, edge weights)
        self.nearest_nodes = nearest_nodes.to(torch.int64)
        self.ablation = ablation
        assert self.ablation in ['None', 'DGLR', 'DGTV', 'UT'], 'ablation should be None, DGLR, DGTV or UT'

        self.u_ew = None # place holder # dict: i: [B, T, k, n_heads]
        self.d_ew = None # place holder
        # iterations
        self.ADMM_iters = ADMM_info['ADMM_iters']# 50
        # self.inner_ADMM_iters = 30
        self.CG_iters = ADMM_info['CG_iters'] # 3
        self.PGD_iters = ADMM_info['PGD_iters']
        # Lagrangian parameters
        self.mu_u_init = ADMM_info['mu_u_init'] #3
        self.mu_d1_init = ADMM_info['mu_d1_init'] #$ 3
        self.mu_d2_init = ADMM_info['mu_d2_init'] # 3
        self.mu_u = Parameter(torch.ones((self.ADMM_iters,), device=self.device) * self.mu_u_init, requires_grad=True)
        if self.ablation == 'None':
            self.mu_d1 = Parameter(torch.ones((self.ADMM_iters,), device=self.device) * self.mu_d1_init, requires_grad=True)
        self.mu_d2 = Parameter(torch.ones((self.ADMM_iters,), device=self.device) * self.mu_d2_init, requires_grad=True)

        # ADMM params, empirical initialized?
        self.rho_init = math.sqrt(self.n_nodes / self.T)
        self.rho_u_init = math.sqrt(self.n_nodes / self.T)
        self.rho_d_init = math.sqrt(self.n_nodes / self.T)

        if self.ablation in ['None', 'DGLR']:
            self.rho = Parameter(torch.ones((self.ADMM_iters,), device=self.device) * self.rho_init, requires_grad=True)

        self.rho_u = Parameter(torch.ones((self.ADMM_iters,), device=self.device) * self.rho_u_init, requires_grad=True)

        if self.ablation != 'DGLR':
            self.rho_d = Parameter(torch.ones((self.ADMM_iters,), device=self.device) * self.rho_d_init, requires_grad=True)
        # CGD params, emperical initialized
        alpha_init = 0.08
        self.alpha_x_init = alpha_init
        self.alpha_zu_init = alpha_init
        self.alpha_zd_init = alpha_init
        self.beta_x_init = alpha_init
        self.beta_zu_init = alpha_init
        self.beta_zd_init = alpha_init
        self.alpha_x = Parameter(torch.ones((self.ADMM_iters, self.CG_iters, self.n_heads, 1), device=self.device) * self.alpha_x_init, requires_grad=True)
        self.beta_x = Parameter(torch.ones((self.ADMM_iters, self.CG_iters, self.n_heads, 1), device=self.device) * self.beta_x_init, requires_grad=True)
        self.alpha_zu = Parameter(torch.ones((self.ADMM_iters, self.CG_iters, self.n_heads, 1), device=self.device) * self.alpha_zu_init, requires_grad=True)
        self.beta_zu = Parameter(torch.ones((self.ADMM_iters, self.CG_iters, self.n_heads, 1), device=self.device) * self.beta_zu_init, requires_grad=True)
        if self.ablation != 'DGLR':
            self.alpha_zd = Parameter(torch.ones((self.ADMM_iters, self.CG_iters, self.n_heads, 1), device=self.device) * self.alpha_zd_init, requires_grad=True)
            self.beta_zd = Parameter(torch.ones((self.ADMM_iters, self.CG_iters, self.n_heads, 1), device=self.device) * self.beta_zd_init, requires_grad=True)
        # PGD params: for now we directly solve phi^{tau+1}
        # self.epsilon_init = 0.1
        # self.epsilon = Parameter(torch.ones((self.ADMM_iters, self.PGD_iters), device=self.device) * self.epsilon_init, requires_grad=True)

        self.comb_weights = Parameter(torch.ones((self.n_heads,), device=self.device) / self.n_heads, requires_grad=True)

    def apply_op_Lu(self, x):
        '''
        Args:
            x in (B, T, n_nodes, n_head, n_channel) # B: batchsize
            edges in (B, T, edges, 2)
            edge_weights in (B, T, N, k, n_heads)
        '''
        B, T = x.size(0), x.size(1)
        # pad x
        pad_x = torch.zeros_like(x[:,:,0], device=self.device).unsqueeze(2)
        pad_x = torch.cat((x, pad_x), dim=2)
        return x - (self.u_ew.unsqueeze(-1) * pad_x[:,:,self.nearest_nodes[:,1:].reshape(-1)].view(B, T, self.n_nodes, -1, self.n_heads, self.n_channels)).sum(3)

    def apply_op_Ldr(self, x):
        '''
        self.d_ew in (B, T-1, k, n_heads)
        '''
        B, T = x.size(0), x.size(1)
        pad_x = torch.zeros_like(x[:,:,0], device=self.device).unsqueeze(2)
        pad_x = torch.cat((x, pad_x), dim=2)
        y = torch.zeros_like(x, device=self.device)
        y[:,1:] = x[:,1:] - (self.d_ew.unsqueeze(-1) * pad_x[:,:-1,self.nearest_nodes.view(-1)].view(B, T-1, self.n_nodes, -1, self.n_heads, self.n_channels)).sum(3)
        return y

    def apply_op_Ldr_T(self, x):
        '''
        x in (B, T, N, n_head, n_channels)
        '''
        # print('x', x.size())
        assert not torch.isnan(x).any(), 'Ldr T x input x has NaN value'
        B, T = x.size(0), x.size(1)
        pad_x = torch.zeros_like(x[:,:,0], device=self.device).unsqueeze(2)
        pad_x = torch.cat((x, pad_x), dim=2)

        ##### CHANGED HERE##############################################
        holder = self.d_ew.unsqueeze(-1) * x[:,1:].unsqueeze(3) # in (B, T-1, N, k, n_heads, n_channels)
        in_features = torch.zeros((B, T-1, self.n_nodes + 1, self.n_heads, self.n_channels), device=self.device)
        index = self.nearest_nodes.reshape(-1).unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(B, T-1, 1, self.n_heads, self.n_channels)
        index[index == -1] = self.n_nodes

        if torch.any(index < 0) or torch.any(index >= in_features.size(2)):
            raise ValueError("Index out of bounds")
        
        in_features.scatter_add(2, index, holder.view(B, T-1, -1, self.n_heads, self.n_channels))
        in_features = in_features[:,:,:-1]
        ############### ORIGINAL Version ###############
        # in_features = (self.d_ew.unsqueeze(-1) * pad_x[:, 1:, self.nearest_nodes.view(-1)].view(B, T-1, self.n_nodes, -1, self.n_heads, self.n_channels)).sum(3) # in (B, T-1, N, n_heads, n_channels)
 ####################################
        y = x.clone()
        y[:,0] = torch.zeros_like(x[:,0])
        y[:,:-1] = y[:,:-1] - in_features
        return y
    
    def apply_op_cLdr(self, x):
        y = self.apply_op_Ldr(x)
        y = self.apply_op_Ldr_T(y)
        return y # x[T] = x[T]
    
    def CG_solver(self, LHS_func, RHS:torch.Tensor, x0:torch.Tensor, ADMM_iters, alpha, beta, args=None):
        '''
        Using Conjugated Gradient Method to solve linear euqations LHS_func(x) = RHS (dim=n_nodes, on n_heads graphs with n_channel signals)
        Args:
            x in (B, T, n_nodes, n_heads, n_channels)
            LHS_func: linear function, args in self.__init__()
        '''
        if x0 is None:
            x0 = RHS.clone()
        if args is None:
            r = RHS - LHS_func(x0, ADMM_iters)
        else:
            r = RHS - LHS_func(x0, args, ADMM_iters)
        
        p = r.clone() # in (B, T, n_nodes, n_head, n_channels)
        for i in range(self.CG_iters):
            if args is None:
                Ap = LHS_func(p, ADMM_iters)
            else:
                Ap = LHS_func(p, args, ADMM_iters)
            
            x0 = x0 + alpha[ADMM_iters, i] * p
            r = r - alpha[ADMM_iters, i] * Ap
            p = r + beta[ADMM_iters, i] * p

        return x0 #, alpha, beta
    
    def LHS_simple_x(self, x,y, iters): # all in one as Eq. 10
        HtHx = x.clone()
        HtHx[:,6:] = torch.zeros_like(x[:,y.size(1):])
        return HtHx + self.mu_u[iters] * self.apply_op_Lu(x) + (self.mu_d2[iters] + self.rho[iters] / 2) * self.apply_op_cLdr(x)
    
    def LHS_x(self, x, y, iters):
        HtHx = x.clone()
        HtHx[:,y.size(1):] = torch.zeros_like(x[:,y.size(1):])
        if self.ablation == 'DGTV':
            output = HtHx + (self.rho_u[iters] + self.rho_d[iters]) / 2 * x
        elif self.ablation == 'None':
            output = HtHx + (self.rho_u[iters] + self.rho_d[iters]) / 2 * x + self.rho[iters] / 2 * self.apply_op_cLdr(x)
        elif self.ablation == 'UT':
            pass # TODO
        elif self.ablation == 'DGLR':
            output = HtHx + self.rho[iters] / 2 * self.apply_op_cLdr(x) + self.rho_u[iters] / 2 * x
        return output
    
    def LHS_zu(self, zu, iters):
        return self.mu_u[iters] * self.apply_op_Lu(zu) + self.rho_u[iters] / 2 * zu
    
    def LHS_zd(self, zd, iters):
        return self.mu_d2[iters] * self.apply_op_cLdr(zd) + self.rho_d[iters] / 2 * zd
    
    def soft_threshold(self, phi, lambda_):
        '''
        return soft(x, lambda_) = sgn(x) *max(|x| - labmda_, 0), lambda_ is a number
        '''
        u = torch.abs(phi) - lambda_
        return torch.sign(phi) * u * (u > 0)
    
    def Phi_PGD(self, phi, x, gamma, ADMM_iters):
        for i in range(self.PGD_iters):
            # phi_old = phi.clone()
            df = gamma + self.rho[ADMM_iters] * (phi - self.apply_op_Ldr(x))
            phi = self.soft_threshold(phi - self.epsilon[ADMM_iters, i] * df, self.epsilon[ADMM_iters, i] * self.mu_d1[ADMM_iters])
            # err = torch.norm(phi - phi_old)
        #     if err < tol:
        #         break
        # print(f'PGD iterations {i}: err = {err}')
        return phi
    
    def phi_direct(self, x, gamma, ADMM_iters):
        '''
        phi^{tau+1} = soft_(mu_d1 / rho) (L^d_r x - gamma / rho)
        '''
        s = self.apply_op_Ldr(x) - gamma / self.rho[ADMM_iters]
        d = self.mu_d1[ADMM_iters] / self.rho[ADMM_iters]
        u = torch.abs(s) - d
        return torch.sign(s) * u * (u > 0)
    
    # def single_loop(self, y):
        # pass

    # def nested_loop(self, y): # not complete
    #     if y.size(1) < self.T:
    #         x = LR_guess(y, self.T, self.device)
    #     else:
    #         x = y[:,0:self.T]
    #         y = y[:,0:self.T]
    #     phi = self.apply_op_Ldr(x)

    #     gamma = torch.ones_like(x) * 0.1
    #     for i in range(self.ADMM_iters):
    #         # initializations
    #         gamma_u, gamma_d = torch.ones_like(x) * 0.1, torch.ones_like(x) * 0.1
    #         zu, zd = x.clone(), x.clone()
    #         # phi = self.apply_op_Ldr(x)
    #         phi_old = phi.clone()
    #         for j in range(self.inner_ADMM_iters):
    #             zu_old, zd_old = zu.clone(), zd.clone()
    #             Hty = torch.zeros_like(x)
    #             Hty[:,0:y.size(1)] = y
    #             RHS_x = self.apply_op_Ldr_T(gamma + self.rho[i] * phi) / 2 + (self.rho_u[i] * zu + self.rho_d[i] * zd) / 2 - (gamma_u + gamma_d) / 2 + Hty
    #             assert not torch.isnan(RHS_x).any(), f'RHS_x has NaN value in loop {i}'
    #             assert not torch.isinf(RHS_x).any() and not torch.isinf(-RHS_x).any(), f'RHS_x has inf value in loop {i}'

    #             x = self.CG_solver(self.LHS_x, RHS_x, x, i, self.alpha_x, self.beta_x, args=y)
    #             assert not torch.isnan(x).any(), f'RHS_x has NaN value in loop {i}'
    #             assert not torch.isinf(x).any() and not torch.isinf(x).any(), f'x has inf value in loop {i}'

    #             RHS_zu = gamma_u / 2 + self.rho_u[i] / 2 * x
    #             zu = self.CG_solver(self.LHS_zu, RHS_zu, zu, i, self.alpha_zu, self.beta_zu)
    #             assert not torch.isnan(RHS_zu).any(), f'RHS_zu has NaN value in loop {i}'
    #             assert not torch.isinf(RHS_zu).any() and not torch.isinf(-RHS_zu).any(), f'RHS_zu has inf value in loop {i}'
    #             assert not torch.isnan(zu).any(), f'zu has NaN value in loop {i}'
    #             assert not torch.isinf(zu).any() and not torch.isinf(-zu).any(), f'zu has inf value in loop {i}'

    #             RHS_zd = gamma_d / 2 + self.rho_d[i] / 2 * x
    #             zd = self.CG_solver(self.LHS_zd, RHS_zd, zd, i, self.alpha_zd, self.beta_zd)
    #             assert not torch.isnan(RHS_zd).any(), f'RHS_zd has NaN value in loop {i}'
    #             assert not torch.isinf(RHS_zd).any() and not torch.isinf(-RHS_zd).any(), f'RHS_zd has inf value in loop {i}'
    #             assert not torch.isnan(zd).any(), f'zd has NaN value in loop {i}'
    #             assert not torch.isinf(zd).any() and not torch.isinf(-zd).any(), f'RHS_zd has inf value in loop {i}'
    #             # udpate gamma_u, gamma_d
    #             gamma_u = gamma_u + self.rho_u[i] * (x - zu)
    #             gamma_d = gamma_d + self.rho_d[i] * (x - zd)
    #             # criterions
    #             primal_residual = max(torch.norm(x - zu), torch.norm(x - zd))
    #             dual_residual = max(torch.norm(-self.rho_u * (zu - zu_old)), torch.norm(-self.rho_d * (zd - zd_old)))
    #             if primal_residual < 1e-4 and dual_residual < 1e-4:
    #                 break
    #         # solve phi
    #         phi = self.phi_direct(x, gamma, i)
    #         gamma = gamma + self.rho[i] * (phi - self.apply_op_Ldr(x))
    #         # criterion
    #         primal_residual = torch.norm(phi - self.apply_op_Ldr(x))
    #         dual_residual = torch.norm(-self.rho[i] * self.apply_op_Ldr_T(phi - phi_old))
    #         print(f'outer ADMM iters {i}: pri_err = {primal_residual}, dual_err = {dual_residual}')
    #         if primal_residual < 1e-3 and dual_residual < 1e-3:
    #             break
    #     print(f'outer ADMM iters {i}') 
    #     return x             
    
    def forward(self, y, mask=None, simple=False):
        '''
        y in (batch, t, n_nodes, signal_channels)
        actually the ADMMBlock accepts x
        '''
        # primal guess x
        if y.size(1) < self.T:# actually not used
            x = LR_guess(y, self.T, self.device)
        else:
            assert mask is not None, 'mask should be t for sequential inputs'
            x = y[:,0:self.T]
            y = y[:,0:mask]
        # multihead
        y = y.unsqueeze(-2).repeat(1,1,1,self.n_heads, 1)
        x = x.unsqueeze(-2).repeat(1,1,1,self.n_heads, 1)

        # print('any NaN in d_ew', torch.isnan(self.d_ew).nonzero(as_tuple=True), self.d_ew.max(), self.d_ew.min())
        # print('any NaN in phi', torch.isnan(phi).any())
        gamma_u, gamma_d = torch.ones_like(x) * 0.05, torch.ones_like(x) * 0.1
        if self.ablation == 'None':
            gamma = torch.ones_like(x) * 0.1
            phi = self.apply_op_Ldr(x)
            
        zu, zd = x.clone(), x.clone()
        
        for i in range(self.ADMM_iters):
            # zu_old, zd_old = zu.clone(), zd.clone()
            # phi_old = phi.clone()
            Hty = torch.zeros_like(x)
            Hty[:,0:y.size(1)] = y
            if simple:
                RHS_x = self.apply_op_Ldr_T(self.rho * phi + gamma) / 2 + Hty
                assert not torch.isnan(RHS_x).any(), f'RHS_x has NaN value in loop {i}, d_ew in {self.d_ew.max().item():.4f}, {self.d_ew.min().item():.4f}'
                assert not torch.isinf(RHS_x).any() and not torch.isinf(-RHS_x).any(), f'RHS_x has inf value in loop {i}'
                x = self.CG_solver(self.LHS_simple_x, RHS_x, x, i, self.alpha_x, self.beta_x)
                assert not torch.isnan(x).any(), f'RHS_x has NaN value in loop {i}'
                assert not torch.isinf(x).any() and not torch.isinf(x).any(), f'x has inf value in loop {i}'
            else:
                # print(torch.isnan(gamma + self.rho[i] * phi).any(), torch.isnan(gamma).any(), )
                if self.ablation == 'DGTV':
                    RHS_x = (self.rho_u[i] * zu + self.rho_d[i] * zd) / 2 - (gamma_u + gamma_d) / 2 + Hty
                elif self.ablation == 'None':
                    RHS_x = self.apply_op_Ldr_T(gamma + self.rho[i] * phi) / 2 + (self.rho_u[i] * zu + self.rho_d[i] * zd) / 2 - (gamma_u + gamma_d) / 2 + Hty
                elif self.ablation == 'UT':
                    pass
                elif self.ablation == 'DGLR':
                    RHS_x = self.apply_op_Ldr_T(gamma + self.rho[i] * phi) / 2 + self.rho_u[i] * zu / 2 - gamma_u / 2 + Hty
                # print(torch.isnan(zu).any(), torch.isnan(zd).any())
                    assert not torch.isnan(gamma + self.rho[i] * phi).any(), f'Ldr T input has NaN in loop {i}, gamma {torch.isnan(gamma).any()}, rho[i] {torch.isnan(self.rho[i]).any()}, phi {torch.isnan(phi).any()}'

                assert not torch.isnan(RHS_x).any(), f'RHS_x has NaN value in loop {i}, d_ew in {self.d_ew.max().item():.4f}, {self.d_ew.min().item():.4f}, NaN {torch.isnan(self.d_ew).any()}, (rho_u, rho_d)[i] has NaN ({torch.isnan(self.rho_u[i]).any()}, {torch.isnan(self.rho_d[i]).any()}), (z_u, z_d) has NaN ({torch.isnan(zu).any()}, {torch.isnan(zd).any()}), (gamma_u, gamma_d) has NaN ({torch.isnan(gamma_u).any()}, {torch.isnan(gamma_d).any()})'
                assert not torch.isinf(RHS_x).any() and not torch.isinf(-RHS_x).any(), f'RHS_x has inf value in loop {i}'
                # print('RHS_x', torch.isnan(RHS_x).any(), RHS_x.max(), RHS_x.min())
                # solve x with zu, zd, update x
                x = self.CG_solver(self.LHS_x, RHS_x, x, i, self.alpha_x, self.beta_x, args=y)
                assert not torch.isnan(x).any(), f'RHS_x has NaN value in loop {i}'
                assert not torch.isinf(x).any() and not torch.isinf(x).any(), f'x has inf value in loop {i}'
                # print('x', torch.isnan(x).any())
                # solve zu, zd with x, update zu, zd
                RHS_zu = gamma_u / 2 + self.rho_u[i] / 2 * x
                zu = self.CG_solver(self.LHS_zu, RHS_zu, zu, i, self.alpha_zu, self.beta_zu)
                assert not torch.isnan(RHS_zu).any(), f'RHS_zu has NaN value in loop {i}'
                assert not torch.isinf(RHS_zu).any() and not torch.isinf(-RHS_zu).any(), f'RHS_zu has inf value in loop {i}'
                # print('RHS_zu, zu', torch.isnan(RHS_zu).any(), RHS_zu.max(), RHS_zu.min(), torch.isnan(zu).any(), zu.max(), zu.min())
                RHS_zd = gamma_d / 2 + self.rho_d[i] / 2 * x
                zd = self.CG_solver(self.LHS_zd, RHS_zd, zd, i, self.alpha_zd, self.beta_zd)
                assert not torch.isnan(RHS_zd).any(), f'RHS_zd has NaN value in loop {i}'
                assert not torch.isinf(RHS_zd).any() and not torch.isinf(-RHS_zd).any(), f'RHS_zd has inf value in loop {i}'

                gamma_u = gamma_u + self.rho_u[i] * (x - zu)
                gamma_d = gamma_d + self.rho_d[i] * (x - zd)
            # udpata phi
            # phi = self.Phi_PGD(phi, x, gamma, i) # 
            if self.ablation == 'None':
                phi = self.phi_direct(x, gamma, i)
                gamma = gamma + self.rho[i] * (phi - self.apply_op_Ldr(x))
                assert not (torch.isnan(gamma).any() or torch.isnan(phi).any()), f'gamma has NaN {torch.isnan(gamma).any()}, phi has NaN {torch.isnan(phi).any()}'
        #     # criterion
        #     primal_residual = max(torch.norm(phi - self.apply_op_Ldr(x)), torch.norm(x - zu), torch.norm(x - zd))
        #     dual_residual = max(torch.norm(-self.rho[i] * self.apply_op_Ldr_T(phi - phi_old)), torch.norm(-self.rho_u[i] * (zu - zu_old)), torch.norm(-self.rho_d[i] * (zd - zd_old)))
        #     print(f'ADMM iters {i}: pri_err = {primal_residual}, dual_err = {dual_residual}')
        #     if primal_residual < 1e-3 and dual_residual < 1e-3:
        #         break
        # print(f'single ADMM iters {i}')
        # combination weights
        #output = self.comb_fc(x.transpose(-2, -1)).squeeze(-1)
        output = torch.einsum('btnhc, h -> btnc', x, self.comb_weights)
        # print('ADMM out', output.max(), output.min())
        return output