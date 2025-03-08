import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

class SystemOfPolynomialEquation:
    def __init__(self, variables, system):
        if len(variables) != len(system):
            raise ValueError("The number of equations must match the number of variables.")
        
        self.variables = variables
        self.system = system  # Keep as a list, not a Matrix

        # Compute Jacobian manually as a list of lists
        self.jacobian_matrix = [[sp.diff(f, var) for var in variables] for f in system]

        # Convert symbolic functions into efficient NumPy functions
        self.system_func = sp.lambdify(self.variables, self.system, modules="numpy")
        self.jacobian_func = sp.lambdify(self.variables, self.jacobian_matrix, modules="numpy")

        self.regularization_term = 1e-6 * np.eye(len(variables))

    def eval(self, values):
        """Evaluate the system at specific values for variables."""
        return np.asarray(self.system_func(*values))

    def jacobian(self, values):
        """Evaluate the Jacobian at specific values for variables."""
        return np.asarray(self.jacobian_func(*values))

    def find_roots_newton_batch(self, guesses, max_iter=100, tol=1e-6):
        """Find roots using Newton's method for all points simultaneously."""
        guesses = np.asarray(guesses)  # (batch_size, num_vars)
        batch_size = guesses.shape[0]
        num_eq = len(self.variables)
        total_distances = np.zeros(batch_size)

        for _ in range(max_iter):
            # Evaluate F at all batch points; shape: (batch_size, num_eq)
            F = np.array(self.system_func(*guesses.T)).T  
            # Reshape F to (batch_size, num_eq, 1) for solve
            F = F.reshape(batch_size, num_eq, 1)

            # Evaluate Jacobian; initial shape from lambdify: (num_eq, num_vars, batch_size)
            J = np.array(self.jacobian_func(*guesses.T))  
            # Rearrange axes so that the batch dimension is first: (batch_size, num_eq, num_vars)
            J = np.moveaxis(J, -1, 0)
            
            # Solve J * delta = F for each batch element
            try:
                delta = np.linalg.solve(J, F)  # Expected shape: (batch_size, num_eq, 1)
            except np.linalg.LinAlgError:
                return np.full(batch_size, np.nan)
            
            # Squeeze delta to shape (batch_size, num_eq)
            delta = delta.reshape(batch_size, num_eq)
            guesses -= delta  # Update guesses

            # Compute Euclidean norm for each delta
            dist = np.linalg.norm(delta, axis=1)
            total_distances += dist

            if np.all(dist < tol):
                break

        return total_distances

# Some good cmap palette are
cmap = 'hot'
res = 1024
degree = 5

degree_p1 = degree + 1
x, y = sp.symbols("x y")

# Generate random coefficients for terms up to degree 4
coeffs1 = np.random.uniform(-2, 2, size=(degree_p1, degree_p1))
coeffs2 = np.random.uniform(-2, 2, size=(degree_p1, degree_p1))
f1 = sum(coeffs1[i, j] * x**i * y**j for i in range(degree) for j in range(degree - i))
f2 = sum(coeffs2[i, j] * x**i * y**j for i in range(degree) for j in range(degree - i))

# f1 = x**3 - 3*x*y**2 + sp.sin(y)
# f2 = y**3 - 3*x**2*y + sp.cos(x)
f1 = x**3 - 3*x*y**2 + sp.exp(-x**2 - y**2) * sp.sin(3*x)
f2 = y**3 - 3*x**2*y + sp.exp(-x**2 - y**2) * sp.cos(3*y)
# f1 = 0.1*x**5 - 2*x*y + x - 2
# f2 = -0.5*y**2 - 3*y**2*x + 1

system = SystemOfPolynomialEquation([x, y], [f1, f2])

# If you want to save current equations
import pickle as pkl
# params = {
#     'vars': [x, y],
#     'funcs': [f1, f2]
# }

# with open('fractal/system_params.pkl', 'wb') as f:
#     pkl.dump(params, f)

# with open('fractal/system_params.pkl', 'rb') as f:
#     params = pkl.load(f)
#     system = SystemOfPolynomialEquation(params['vars'], params['funcs'])

# Grid setup
grid_size = res
x_vals = np.linspace(-2.5, 2.5, grid_size)
y_vals = np.linspace(-2.5, 2.5, grid_size)

# Create meshgrid and stack into a batch of initial guesses (shape: (grid_size^2, 2))
X, Y = np.meshgrid(x_vals, y_vals, indexing="ij")
initial_guesses = np.column_stack([X.ravel(), Y.ravel()])

# Compute travel distances in batch and reshape back to grid
travel_map = system.find_roots_newton_batch(initial_guesses, max_iter=100, tol=1e-6)
travel_map = travel_map.reshape(grid_size, grid_size)

# Normalize travel distance for brightness control
for _ in range(6):
    travel_map = np.log1p(travel_map)

travel_map = (travel_map - np.min(travel_map)) / (np.max(travel_map) - np.min(travel_map))

# Plot the fractal
fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(travel_map, extent=(-2, 2, -2, 2), origin='lower', cmap=cmap, vmin=0, vmax=1)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Newton Fractal - Root Convergence and Travel Distance")
plt.show()

plt.imsave(r'fractal\newton_mult_test' + cmap + str(res) + '.png', np.flipud(travel_map), cmap=cmap, vmin=0, vmax=1)
