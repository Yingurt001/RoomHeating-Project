"""
Unified plotting utilities for the room heating project.

Provides consistent, publication-quality figures for all experiments.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_temperature_and_heater(t, T, heater, title="Temperature Control",
                                T_set=20.0, T_a=5.0, delta=None,
                                save_path=None):
    """
    Standard dual-panel plot: temperature + heater state.

    Parameters
    ----------
    t : ndarray
        Time array.
    T : ndarray
        Temperature array.
    heater : ndarray
        Heater state (0/1).
    title : str
        Figure title.
    T_set : float
        Setpoint temperature.
    T_a : float
        Ambient temperature.
    delta : float, optional
        Hysteresis band half-width.
    save_path : str, optional
        Path to save figure.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(t, T, 'b-', linewidth=1.5, label=r'$T(t)$')
    ax1.axhline(y=T_set, color='r', linestyle='--', alpha=0.7,
                label=f'$T_{{set}} = {T_set}$°C')
    ax1.axhline(y=T_a, color='cyan', linestyle=':', alpha=0.7,
                label=f'$T_a = {T_a}$°C')

    if delta is not None and delta > 0:
        ax1.axhline(y=T_set + delta, color='orange', linestyle='--', alpha=0.5,
                     label=f'Hysteresis ±{delta}°C')
        ax1.axhline(y=T_set - delta, color='orange', linestyle='--', alpha=0.5)
        ax1.axhspan(T_set - delta, T_set + delta, alpha=0.08, color='orange')

    ax1.set_ylabel('Temperature (°C)', fontsize=12)
    ax1.set_title(title, fontsize=14)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.fill_between(t, heater, step='post', alpha=0.4, color='red',
                     label='Heater ON')
    ax2.step(t, heater, 'r-', linewidth=1, where='post')
    ax2.set_xlabel('Time (minutes)', fontsize=12)
    ax2.set_ylabel('Heater', fontsize=12)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['OFF', 'ON'])
    ax2.set_ylim(-0.1, 1.3)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_temperature_continuous(t, T, u, title="Continuous Control",
                                T_set=20.0, T_a=5.0, U_max=15.0,
                                save_path=None):
    """
    Dual-panel plot for continuous controllers (PID, LQR, Pontryagin).
    Shows temperature and continuous control signal.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(t, T, 'b-', linewidth=1.5, label=r'$T(t)$')
    ax1.axhline(y=T_set, color='r', linestyle='--', alpha=0.7,
                label=f'$T_{{set}} = {T_set}$°C')
    ax1.axhline(y=T_a, color='cyan', linestyle=':', alpha=0.7,
                label=f'$T_a = {T_a}$°C')
    ax1.set_ylabel('Temperature (°C)', fontsize=12)
    ax1.set_title(title, fontsize=14)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.plot(t, u, 'r-', linewidth=1.2, label='Control $u(t)$')
    ax2.axhline(y=U_max, color='gray', linestyle=':', alpha=0.5,
                label=f'$U_{{max}} = {U_max}$')
    ax2.set_xlabel('Time (minutes)', fontsize=12)
    ax2.set_ylabel('Control Input', fontsize=12)
    ax2.set_ylim(-0.5, U_max * 1.1)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_comparison(results_dict, T_set=20.0, save_path=None):
    """
    Overlay temperature curves from multiple controllers on one plot.

    Parameters
    ----------
    results_dict : dict
        {name: (t, T, u)} for each controller.
    T_set : float
        Setpoint.
    save_path : str, optional
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})

    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))

    for (name, (t, T, u)), color in zip(results_dict.items(), colors):
        ax1.plot(t, T, linewidth=1.5, label=name, color=color)
        ax2.plot(t, u, linewidth=1.2, label=name, color=color)

    ax1.axhline(y=T_set, color='k', linestyle='--', alpha=0.5,
                label=f'$T_{{set}} = {T_set}$°C')
    ax1.set_ylabel('Temperature (°C)', fontsize=12)
    ax1.set_title('Controller Comparison', fontsize=14)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Time (minutes)', fontsize=12)
    ax2.set_ylabel('Control Input', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_metrics_bar(metrics_dict, save_path=None):
    """
    Bar chart comparing metrics across controllers.

    Parameters
    ----------
    metrics_dict : dict
        {controller_name: metrics_dict} where each inner dict has
        keys like 'energy', 'rmse', 'settling_time', etc.
    """
    names = list(metrics_dict.keys())
    metric_keys = ['energy', 'rmse', 'max_overshoot', 'settling_time',
                   'switching_count', 'unified_cost']
    metric_labels = ['Energy', 'RMSE (°C)', 'Max Overshoot (°C)',
                     'Settling Time (min)', 'Switches', 'Unified Cost J']

    n_metrics = len(metric_keys)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()

    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))

    for i, (key, label) in enumerate(zip(metric_keys, metric_labels)):
        ax = axes[i]
        values = [metrics_dict[name].get(key, 0) for name in names]
        bars = ax.bar(names, values, color=colors)
        ax.set_title(label, fontsize=12)
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Controller Performance Metrics', fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_pareto(metrics_dict, x_key='energy', y_key='rmse',
                x_label='Energy Consumption', y_label='Temperature RMSE (°C)',
                save_path=None):
    """
    Pareto front plot: X = energy, Y = RMSE.

    Parameters
    ----------
    metrics_dict : dict
        {label: metrics_dict}.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    for name, m in metrics_dict.items():
        ax.scatter(m[x_key], m[y_key], s=100, label=name, zorder=5)
        ax.annotate(name, (m[x_key], m[y_key]), fontsize=9,
                    xytext=(5, 5), textcoords='offset points')

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title('Pareto Front: Energy vs Comfort', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_1d_heatmap(t, x, T_field, title="1D Temperature Field",
                    T_set=20.0, save_path=None):
    """
    Heatmap of T(x, t) for 1D PDE model.

    Parameters
    ----------
    t : ndarray of shape (nt,)
    x : ndarray of shape (nx,)
    T_field : ndarray of shape (nx, nt)
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    im = ax.pcolormesh(t, x, T_field, shading='auto', cmap='hot')
    cb = plt.colorbar(im, ax=ax, label='Temperature (°C)')
    ax.set_xlabel('Time (minutes)', fontsize=12)
    ax.set_ylabel('Position x (m)', fontsize=12)
    ax.set_title(title, fontsize=14)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_2d_snapshots(x, y, T_field_3d, times, t_array, title_prefix="2D Room",
                      T_set=20.0, save_path=None):
    """
    Plot 2D temperature field snapshots at selected times.

    Parameters
    ----------
    x, y : ndarray
        Spatial coordinates.
    T_field_3d : ndarray of shape (nx, ny, nt)
    times : list of float
        Times at which to take snapshots.
    t_array : ndarray
        Full time array.
    """
    n_snaps = len(times)
    fig, axes = plt.subplots(1, n_snaps, figsize=(5 * n_snaps, 4))
    if n_snaps == 1:
        axes = [axes]

    X, Y = np.meshgrid(x, y, indexing='ij')

    for ax, t_snap in zip(axes, times):
        idx = np.argmin(np.abs(t_array - t_snap))
        T_snap = T_field_3d[:, :, idx]

        im = ax.pcolormesh(X, Y, T_snap, shading='auto', cmap='hot',
                           vmin=T_field_3d.min(), vmax=T_field_3d.max())
        ax.set_title(f't = {t_array[idx]:.1f} min', fontsize=11)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(f'{title_prefix} Temperature Field', fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close(fig)
    return fig
