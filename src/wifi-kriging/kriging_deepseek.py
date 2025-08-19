import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import curve_fit

def generate_sample_data(num_points=100, x_range=(0, 100), y_range=(0, 100), seed=None):
    """
    Generate random sample points in 2D space with positive values.
    
    Parameters:
    num_points: Number of sample points to generate
    x_range: Tuple defining the x-axis range
    y_range: Tuple defining the y-axis range
    seed: Random seed for reproducibility
    
    Returns:
    points: Array of shape (num_points, 2) containing (x, y) coordinates
    values: Array of shape (num_points,) containing positive values
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random points
    x_coords = np.random.uniform(x_range[0], x_range[1], num_points)
    y_coords = np.random.uniform(y_range[0], y_range[1], num_points)
    points = np.column_stack((x_coords, y_coords))
    
    # Generate positive values with some spatial correlation
    # Using a combination of sinusoidal patterns to create spatial structure
    x_norm = (x_coords - x_range[0]) / (x_range[1] - x_range[0])
    y_norm = (y_coords - y_range[0]) / (y_range[1] - y_range[0])
    
    values = (np.sin(2 * np.pi * x_norm) * np.cos(3 * np.pi * y_norm) + 
             0.5 * np.sin(5 * np.pi * x_norm) + 0.5 * np.cos(7 * np.pi * y_norm))
    
    # Scale and shift to ensure all values are positive
    values = (values - np.min(values)) * 10 + 1
    
    return points, values

def calculate_empirical_variogram(points, values, num_bins=20):
    """
    Calculate the empirical variogram from sample data.
    
    Parameters:
    points: Array of shape (n, 2) containing sample coordinates
    values: Array of shape (n,) containing sample values
    num_bins: Number of bins for distance grouping
    
    Returns:
    bin_centers: Array of bin center distances
    gamma: Array of variogram values for each bin
    """
    # Calculate all pairwise distances
    dist_matrix = cdist(points, points)
    
    # Calculate all pairwise value differences
    value_diffs = np.abs(values[:, np.newaxis] - values)
    
    # Flatten matrices and remove duplicates and zero distances
    dists = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
    diffs_squared = value_diffs[np.triu_indices_from(value_diffs, k=1)] ** 2
    
    # Bin the distances and calculate variogram values
    max_dist = np.max(dists)
    bins = np.linspace(0, max_dist, num_bins + 1)
    bin_indices = np.digitize(dists, bins) - 1
    
    # Calculate mean squared difference for each bin
    gamma = np.zeros(num_bins)
    counts = np.zeros(num_bins)
    
    for i in range(num_bins):
        mask = (bin_indices == i)
        if np.any(mask):
            gamma[i] = np.mean(diffs_squared[mask])
            counts[i] = np.sum(mask)
    
    # Calculate bin centers
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Filter out bins with no data points
    valid_bins = counts > 0
    return bin_centers[valid_bins], gamma[valid_bins]

def exponential_variogram_model(h, nugget, sill, range_param):
    """
    Exponential variogram model.
    
    Parameters:
    h: Distance
    nugget: Nugget effect (discontinuity at origin)
    sill: Sill value (plateau of the variogram)
    range_param: Range parameter (distance at which 95% of sill is reached)
    
    Returns:
    Variogram value at distance h
    """
    return nugget + sill * (1 - np.exp(-3 * h / range_param))

def fit_variogram_model(bin_centers, gamma):
    """
    Fit a theoretical variogram model to empirical data.
    
    Parameters:
    bin_centers: Distances from empirical variogram
    gamma: Variogram values from empirical variogram
    
    Returns:
    Function of the fitted variogram model
    params: Fitted parameters (nugget, sill, range)
    """
    # Initial parameter guesses
    initial_guess = [0, np.max(gamma), np.max(bin_centers) / 2]
    
    # Fit the model
    params, _ = curve_fit(exponential_variogram_model, bin_centers, gamma, p0=initial_guess, maxfev=5000)
    
    # Create a function with the fitted parameters
    def fitted_model(h):
        return exponential_variogram_model(h, *params)
    
    return fitted_model, params

def ordinary_kriging(points, values, grid_x, grid_y, variogram_model):
    """
    Perform ordinary kriging interpolation.
    
    Parameters:
    points: Sample points coordinates
    values: Sample values
    grid_x: X coordinates of interpolation grid
    grid_y: Y coordinates of interpolation grid
    variogram_model: Fitted variogram model function
    
    Returns:
    kriged_grid: Interpolated values on the grid
    """
    n_samples = points.shape[0]
    n_grid = grid_x.shape[0] * grid_x.shape[1]
    
    # Create grid coordinates
    grid_coords = np.column_stack((grid_x.flatten(), grid_y.flatten()))
    
    # Initialize kriging result
    kriged_values = np.zeros(n_grid)
    
    # For each grid point, solve the kriging system
    for i, grid_point in enumerate(grid_coords):
        # Calculate distances between sample points
        sample_dists = cdist(points, points)
        sample_variogram = variogram_model(sample_dists)
        
        # Calculate distances from sample points to grid point
        point_dists = cdist(points, [grid_point])
        point_variogram = variogram_model(point_dists)
        
        # Construct the kriging matrix
        kriging_matrix = np.ones((n_samples + 1, n_samples + 1))
        kriging_matrix[:n_samples, :n_samples] = sample_variogram
        kriging_matrix[n_samples, :n_samples] = 1
        kriging_matrix[:n_samples, n_samples] = 1
        kriging_matrix[n_samples, n_samples] = 0  # Lagrange multiplier
        
        # Construct the right-hand side vector
        rhs = np.ones(n_samples + 1)
        rhs[:n_samples] = point_variogram.flatten()
        rhs[n_samples] = 1  # Constraint for unbiasedness
        
        # Solve the kriging system
        try:
            weights = np.linalg.solve(kriging_matrix, rhs)
            kriged_values[i] = np.sum(weights[:n_samples] * values)
        except np.linalg.LinAlgError:
            # If matrix is singular, use nearest neighbor
            dists = np.linalg.norm(points - grid_point, axis=1)
            kriged_values[i] = values[np.argmin(dists)]
    
    # Reshape to grid
    kriged_grid = kriged_values.reshape(grid_x.shape)
    
    return kriged_grid

def plot_results(points, values, grid_x, grid_y, kriged_grid, bin_centers, gamma, variogram_model, params):
    """
    Plot the input data, kriging results, and variogram.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot sample points
    scatter = axes[0, 0].scatter(points[:, 0], points[:, 1], c=values, cmap='viridis', s=50, edgecolors='black')
    axes[0, 0].set_title('Sample Points')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    plt.colorbar(scatter, ax=axes[0, 0])
    
    # Plot kriging results
    im = axes[0, 1].imshow(kriged_grid, extent=(grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()), 
                          origin='lower', cmap='viridis', aspect='auto')
    axes[0, 1].scatter(points[:, 0], points[:, 1], c='red', s=20, edgecolors='black')
    axes[0, 1].set_title('Kriging Interpolation')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Y')
    plt.colorbar(im, ax=axes[0, 1])
    
    # Plot empirical and fitted variogram
    h = np.linspace(0, np.max(bin_centers), 100)
    axes[1, 0].plot(bin_centers, gamma, 'bo', label='Empirical Variogram')
    axes[1, 0].plot(h, variogram_model(h), 'r-', label='Fitted Model')
    axes[1, 0].set_xlabel('Distance')
    axes[1, 0].set_ylabel('Semivariance')
    axes[1, 0].set_title('Variogram')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Display variogram parameters
    textstr = '\n'.join((
        f'Nugget: {params[0]:.2f}',
        f'Sill: {params[1]:.2f}',
        f'Range: {params[2]:.2f}'))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axes[1, 0].text(0.05, 0.95, textstr, transform=axes[1, 0].transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
    
    # Plot 3D surface of kriging results
    from mpl_toolkits.mplot3d import Axes3D
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    surf = ax.plot_surface(grid_x, grid_y, kriged_grid, cmap='viridis', alpha=0.8)
    ax.scatter(points[:, 0], points[:, 1], values, c='red', s=50, edgecolors='black')
    ax.set_title('3D Kriging Surface')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Value')
    fig.colorbar(surf, ax=ax, shrink=0.5)
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Generate sample data
    points, values = generate_sample_data(num_points=100, x_range=(0, 100), y_range=(0, 100), seed=42)
    
    # Calculate empirical variogram
    bin_centers, gamma = calculate_empirical_variogram(points, values, num_bins=15)
    
    # Fit variogram model
    variogram_model, params = fit_variogram_model(bin_centers, gamma)
    
    # Create interpolation grid
    grid_x, grid_y = np.meshgrid(np.linspace(0, 100, 50), np.linspace(0, 100, 50))
    
    # Perform kriging
    kriged_grid = ordinary_kriging(points, values, grid_x, grid_y, variogram_model)
    
    # Plot results
    plot_results(points, values, grid_x, grid_y, kriged_grid, bin_centers, gamma, variogram_model, params)