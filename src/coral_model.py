import torch
import torch.nn as nn
import torchcde

class SpatialVectorField(nn.Module):
    def __init__(self, num_sites, hidden_channels, adj_matrix):
        super().__init__()
        self.num_sites = num_sites
        self.hidden_channels = hidden_channels
        self.adj = adj_matrix
        self.gcn_linear = nn.Linear(hidden_channels, hidden_channels)
        self.time_linear = nn.Linear(hidden_channels, hidden_channels)
        self.relu = nn.ReLU()

    def forward(self, z_graph):
        # z_graph shape: (Sites, Hidden)
        # Message Passing: (Sites, Sites) @ (Sites, Hidden)
        z_neighbor = torch.matmul(self.adj, z_graph) 
        z_out = self.gcn_linear(z_neighbor) + self.time_linear(z_graph)
        z_out = self.relu(z_out)
        return z_out

class CoralSTGNCDE(nn.Module):
    def __init__(self, num_sites, input_features, hidden_dim, output_features, adj_matrix):
        super().__init__()
        self.num_sites = num_sites
        self.hidden_dim = hidden_dim
        self.input_features = input_features
        self.adj_matrix = nn.Parameter(adj_matrix, requires_grad=False)
        
        # 1. Encoder
        self.encoder = nn.Linear(input_features, hidden_dim)
        
        # 2. Vector Field
        self.func = SpatialVectorField(num_sites, hidden_dim, self.adj_matrix)
        
        # 3. Projector
        self.projector = nn.Linear(hidden_dim, hidden_dim * input_features)
        
        # 4. Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, output_features),
            nn.Sigmoid() 
        )
        
        # Pre-allocate identity matrix for block-diagonal construction
        # This speeds up the cde_func significantly
        self.register_buffer('site_identity', torch.eye(num_sites).view(num_sites, num_sites, 1, 1))

    def forward(self, coeffs):
        X = torchcde.CubicSpline(coeffs)
        
        # --- INITIAL STATE ---
        # Evaluate at t=0. Shape is (Sites * Features)
        x0_flat = X.evaluate(X.interval[0]) 
        
        # Reshape to (Sites, Features)
        x0 = x0_flat.reshape(self.num_sites, self.input_features)
        
        # Encode -> (Sites, Hidden)
        z0 = self.encoder(x0) 
        
        # Flatten for Solver -> (Sites * Hidden)
        z0_flat = z0.view(-1) 
        
        # --- CDE DYNAMICS ---
        def cde_func(t, z):
            # z is flattened (Sites * Hidden)
            
            # 1. Reshape z to graph: (Sites, Hidden)
            z_graph = z.view(self.num_sites, self.hidden_dim)
            
            # 2. Evolve z spatially (GCN)
            h = self.func(z_graph) # (Sites, Hidden)
            
            # 3. Project to control sensitivity (The 'Matrix')
            # Each site generates a (Hidden x Input) sensitivity block
            sensitivity = self.projector(h) # (Sites, Hidden * Inputs)
            
            # Reshape: (Sites, Hidden, Inputs)
            sens_blocks = sensitivity.view(self.num_sites, self.hidden_dim, self.input_features)
            
            # 4. Construct Block Diagonal Matrix
            # We need a matrix of shape (Sites*Hidden, Sites*Inputs)
            # where the diagonal blocks are 'sens_blocks' and off-diagonals are zero.
            # Trick: Broadcast multiply Identity with Blocks
            # (Sites, Sites, 1, 1) * (Sites, 1, Hidden, Inputs) -> (Sites, Sites, Hidden, Inputs)
            # The broadcast matches the first 'Sites' dim of identity with 'Sites' dim of blocks
            
            # Note: We align "Row Site" with "Col Site" via the diagonal
            matrix_4d = self.site_identity * sens_blocks.unsqueeze(1)
            
            # Reshape to 2D Matrix: (Sites*Hidden, Sites*Inputs)
            # We permute to (Site_Row, Hidden, Site_Col, Input) before flattening
            matrix_2d = matrix_4d.permute(0, 2, 1, 3).reshape(
                self.num_sites * self.hidden_dim, 
                self.num_sites * self.input_features
            )
            
            return matrix_2d

        # Solve Integration
        z_T = torchcde.cdeint(X=X, func=cde_func, z0=z0_flat, t=X.grid_points)
        
        # Reshape output: (Time, Sites, Hidden)
        time_steps = z_T.shape[0]
        z_T_spatial = z_T.view(time_steps, self.num_sites, self.hidden_dim)
        
        # Decode
        prediction = self.decoder(z_T_spatial)
        return prediction