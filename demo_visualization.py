"""
Demonstration of PINN with comprehensive visualization.

This script shows how to use the improved PINN implementation with
the visualization utilities to create publication-quality plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

# Import our improved PINN implementations
from pinn_raissi_improved import (
    ContinuousPINN, MLP, BurgersConfig, TrainConfig, 
    burgers_residual, u0_true, demo
)

# Import visualization utilities
from visualization import PINNVisualizer, quick_plot_1d, quick_plot_loss


def analytical_burgers_solution(t: np.ndarray, x: np.ndarray, nu: float) -> np.ndarray:
    """
    Approximate analytical solution for Burgers equation with given IC/BC.
    
    This is a simplified approximation for demonstration purposes.
    For the exact solution, numerical methods or more complex analytical
    expressions would be needed.
    """
    # Simple decay approximation
    return -np.sin(np.pi * x) * np.exp(-nu * np.pi**2 * t)


def comprehensive_demo_with_visualization():
    """
    Run a comprehensive PINN demonstration with extensive visualization.
    """
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    cfg = BurgersConfig()
    viz = PINNVisualizer(figsize=(12, 8))
    
    # Create and train PINN
    print("Setting up and training PINN...")
    model = MLP(in_dim=2, hidden_layers=8, width=64, out_dim=1)
    pinn = ContinuousPINN(
        model=model,
        device=device,
        pde_residual_fn=burgers_residual,
        cfg_burgers=cfg,
    )
    
    # Training configuration
    tcfg = TrainConfig(
        n_u0=200,
        n_bc=200,
        n_f=20_000,
    # Training configuration
    tcfg = TrainConfig(
        n_u0=200,
        n_bc=200,
        n_f=20_000,
        lr=1e-3,
        adam_steps=12_000,
        lbfgs_max_iter=300,
        seed=123,
        track_losses=True,
        print_frequency=2000
    )
    
    # Train the PINN
    loss_history = pinn.train(tcfg, weights=(1.0, 1.0, 1.0))
    
    # Create evaluation grid
    print("Generating predictions...")
    nt, nx = 51, 101
    t_eval = np.linspace(cfg.tmin, cfg.tmax, nt, dtype=np.float32)
    x_eval = np.linspace(cfg.xmin, cfg.xmax, nx, dtype=np.float32)
    TT, XX = np.meshgrid(t_eval, x_eval, indexing="ij")
    
    # Get PINN predictions
    U_pred = pinn.predict(TT, XX)
    
    # Generate analytical solution for comparison
    U_analytical = analytical_burgers_solution(TT, XX, cfg.nu)
    
    # 1. Plot training loss curves
    print("Creating loss curve visualization...")
    fig_loss = viz.plot_loss_curves(
        loss_history=loss_history,
        title="PINN Training Loss History",
        log_scale=True,
        save_path="burgers_loss_curves.png"
    )
    plt.show()
    
    # 2. Plot spacetime evolution
    print("Creating spacetime evolution plot...")
    fig_spacetime = viz.plot_spacetime_1d(
        t=TT, x=XX, u=U_pred,
        title="Burgers Equation: Spacetime Evolution (PINN)",
        xlabel="x", ylabel="t", colorbar_label="u(t,x)",
        save_path="burgers_spacetime_pinn.png"
    )
    plt.show()
    
    # 3. Plot spacetime evolution of analytical solution
    fig_spacetime_analytical = viz.plot_spacetime_1d(
        t=TT, x=XX, u=U_analytical,
        title="Burgers Equation: Spacetime Evolution (Analytical)",
        xlabel="x", ylabel="t", colorbar_label="u(t,x)",
        save_path="burgers_spacetime_analytical.png"
    )
    plt.show()
    
    # 4. Solution comparison at different times
    print("Creating solution comparison plots...")
    time_snapshots = [0.0, 0.2, 0.5, 0.8, 1.0]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, t_snap in enumerate(time_snapshots):
        if i >= len(axes):
            break
            
        # Find closest time index
        t_idx = np.argmin(np.abs(t_eval - t_snap))
        
        # Extract solutions at this time
        u_pred_snap = U_pred[t_idx, :]
        u_analytical_snap = U_analytical[t_idx, :]
        
        axes[i].plot(x_eval, u_analytical_snap, 'r-', linewidth=2, 
                    label='Analytical', alpha=0.8)
        axes[i].plot(x_eval, u_pred_snap, 'b--', linewidth=2, 
                    label='PINN', alpha=0.8)
        axes[i].set_title(f't = {t_snap:.1f}', fontsize=12)
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('u(t,x)')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        
        # Calculate and display error
        rmse = np.sqrt(np.mean((u_pred_snap - u_analytical_snap)**2))
        axes[i].text(0.05, 0.95, f'RMSE: {rmse:.3e}', 
                    transform=axes[i].transAxes, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Remove empty subplot
    if len(time_snapshots) < len(axes):
        fig.delaxes(axes[-1])
    
    plt.suptitle('Burgers Equation: Solution Evolution Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig('burgers_solution_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. Comprehensive error analysis at final time
    print("Creating error analysis...")
    t_final_idx = -1  # Last time step
    u_pred_final = U_pred[t_final_idx, :]
    u_true_final = U_analytical[t_final_idx, :]
    
    fig_error = viz.plot_error_analysis(
        x=x_eval,
        u_pred=u_pred_final,
        u_true=u_true_final,
        title=f"Error Analysis at t = {t_eval[t_final_idx]:.1f}",
        save_path="burgers_error_analysis.png"
    )
    plt.show()
    
    # 6. PDE Residual Analysis
    print("Analyzing PDE residuals...")
    
    # Generate residual evaluation points
    n_residual_points = 5000
    t_residual = np.random.uniform(cfg.tmin, cfg.tmax, (n_residual_points, 1)).astype(np.float32)
    x_residual = np.random.uniform(cfg.xmin, cfg.xmax, (n_residual_points, 1)).astype(np.float32)
    
    # Compute residuals
    import torch
    model.eval()
    with torch.no_grad():
        t_tensor = torch.from_numpy(t_residual).to(device).requires_grad_(True)
        x_tensor = torch.from_numpy(x_residual).to(device).requires_grad_(True)
        residuals = burgers_residual(model, t_tensor, x_tensor, cfg.nu)
        residuals_np = residuals.cpu().numpy()
    
    # Plot residuals
    fig_residuals = viz.plot_residuals(
        x=(x_residual.ravel(), t_residual.ravel()),  # 2D coordinates
        residuals=residuals_np,
        residual_names=["Burgers PDE Residual"],
        title="PDE Residual Distribution",
        save_path="burgers_residuals.png"
    )
    plt.show()
    
    # 7. Create a comparison plot with multiple methods
    print("Creating method comparison...")
    
    # For demonstration, we'll compare PINN with analytical solution
    # In practice, you might compare different PINN architectures or methods
    comparison_results = {
        "PINN Prediction": {"solution": u_pred_final},
        "Analytical Solution": {"solution": u_true_final},
    }
    
    fig_comparison = viz.create_comparison_plot(
        results=comparison_results,
        x=x_eval,
        title="Method Comparison at Final Time",
        save_path="burgers_method_comparison.png"
    )
    plt.show()
    
    # 8. Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    # Overall error metrics
    mae_overall = np.mean(np.abs(U_pred - U_analytical))
    rmse_overall = np.sqrt(np.mean((U_pred - U_analytical)**2))
    max_error = np.max(np.abs(U_pred - U_analytical))
    
    print(f"Overall MAE:        {mae_overall:.6f}")
    print(f"Overall RMSE:       {rmse_overall:.6f}")
    print(f"Maximum Error:      {max_error:.6f}")
    print(f"Final Training Loss: {loss_history['total'][-1]:.2e}")
    
    # Residual statistics
    residual_mean = np.mean(np.abs(residuals_np))
    residual_max = np.max(np.abs(residuals_np))
    residual_std = np.std(residuals_np)
    
    print(f"\nPDE Residual Mean:  {residual_mean:.6f}")
    print(f"PDE Residual Max:   {residual_max:.6f}")
    print(f"PDE Residual Std:   {residual_std:.6f}")
    
    print("\nVisualization files saved:")
    print("- burgers_loss_curves.png")
    print("- burgers_spacetime_pinn.png")
    print("- burgers_spacetime_analytical.png")
    print("- burgers_solution_comparison.png")
    print("- burgers_error_analysis.png")
    print("- burgers_residuals.png")
    print("- burgers_method_comparison.png")


def quick_demo():
    """
    Quick demonstration using the convenience functions.
    """
    print("Running quick visualization demo...")
    
    # Generate some sample data
    x = np.linspace(-1, 1, 100)
    u_true = -np.sin(np.pi * x)
    u_pred = u_true + 0.1 * np.random.normal(0, 1, x.shape)  # Add some noise
    
    # Quick 1D plot
    fig1 = quick_plot_1d(x, u_pred, u_true, "Quick 1D Comparison")
    plt.show()
    
    # Quick loss plot
    fake_loss_history = {
        "total": [1e-1 * (0.95)**i for i in range(1000)],
        "pde": [5e-2 * (0.96)**i for i in range(1000)],
        "ic": [3e-2 * (0.94)**i for i in range(1000)],
        "bc": [2e-2 * (0.97)**i for i in range(1000)]
    }
    
    fig2 = quick_plot_loss(fake_loss_history, "Quick Loss Curves")
    plt.show()


def navier_stokes_visualization_example():
    """
    Example of how to visualize Navier-Stokes results (2D vector fields).
    """
    print("Creating Navier-Stokes visualization example...")
    
    # Generate sample velocity field data (Taylor-Green vortex-like)
    x = np.linspace(0, 2*np.pi, 50)
    y = np.linspace(0, 2*np.pi, 50)
    X, Y = np.meshgrid(x, y)
    
    t = 0.5
    nu = 0.01
    
    # Taylor-Green vortex velocity components
    U = np.sin(X) * np.cos(Y) * np.exp(-2*nu*t)
    V = -np.cos(X) * np.sin(Y) * np.exp(-2*nu*t)
    
    # Pressure field (approximate)
    P = -(np.cos(2*X) + np.cos(2*Y)) * 0.25 * np.exp(-4*nu*t)
    
    viz = PINNVisualizer()
    
    # Plot velocity magnitude
    velocity_magnitude = np.sqrt(U**2 + V**2)
    fig1 = viz.plot_2d_field(
        X, Y, velocity_magnitude,
        title="Velocity Magnitude", 
        colorbar_label="|u|",
        save_path="navier_stokes_velocity_magnitude.png"
    )
    plt.show()
    
    # Plot vector field
    fig2 = viz.plot_vector_field_2d(
        X, Y, U, V,
        title="Velocity Vector Field",
        skip=3,  # Show every 3rd arrow for clarity
        save_path="navier_stokes_vector_field.png"
    )
    plt.show()
    
    # Plot pressure field
    fig3 = viz.plot_2d_field(
        X, Y, P,
        title="Pressure Field",
        colorbar_label="p",
        cmap="RdBu_r",
        save_path="navier_stokes_pressure.png"
    )
    plt.show()


if __name__ == "__main__":
    print("PINN Visualization Demonstration")
    print("="*50)
    
    # Choose which demo to run
    print("\nChoose demonstration:")
    print("1. Comprehensive Burgers equation demo")
    print("2. Quick visualization demo")
    print("3. Navier-Stokes visualization example")
    print("4. All demonstrations")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "1":
        comprehensive_demo_with_visualization()
    elif choice == "2":
        quick_demo()
    elif choice == "3":
        navier_stokes_visualization_example()
    elif choice == "4":
        print("\nRunning all demonstrations...\n")
        comprehensive_demo_with_visualization()
        print("\n" + "-"*50 + "\n")
        quick_demo()
        print("\n" + "-"*50 + "\n")
        navier_stokes_visualization_example()
    else:
        print("Invalid choice. Running comprehensive demo...")
        comprehensive_demo_with_visualization()
    
    print("\nDemonstration completed!")
