# Week 1 Report: 0D Bang-Bang Control Analytical Analysis

## 1. Model

Newton's Law of Cooling (0D ODE):

```
dT/dt = -k(T(t) - T_a) + u(t)
```

### Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| T_a | 5.0 °C | Outdoor temperature |
| T_set | 20.0 °C | Set-point temperature |
| k | 0.1 1/min | Cooling constant |
| U_max | 15.0 °C/min | Max heating rate |
| T_ss | 155.0 °C | Steady-state (heater always ON): T_a + U_max/k |

## 2. Analytical Solution (ODE Derivation)

Rewrite the ODE:

```
dT/dt + kT = kT_a + u(t)
```

Multiply both sides by integrating factor `e^{kt}`:

```
d/dt [e^{kt} T] = (kT_a + u) e^{kt}
```

Integrate from t0 to t:

```
e^{kt} T(t) - e^{kt0} T(t0) = (T_a + u/k)(e^{kt} - e^{kt0})
```

Solve for T(t) — **boxed formula**:

```
┌─────────────────────────────────────────────────────────────┐
│ T(t) = T_a + u/k + e^{-k(t-t0)} · (T(t0) - T_a - u/k)    │
└─────────────────────────────────────────────────────────────┘
```

- **Heater OFF** (u=0): `T(t) = T_a + (T(t0) - T_a) · exp(-k(t-t0))` — decays to T_a
- **Heater ON** (u=U_max): `T(t) = T_ss + (T(t0) - T_ss) · exp(-k(t-t0))` — rises to T_ss

See: `Result/fig1_heating_cooling_curves.png`

## 3. Step 1: Half-Band Switching (T_set ± δ → T_set)

Define hysteresis band: T_H = T_set + δ, T_L = T_set - δ.

In Step 1 (following the Handnote), we compute switching times for **half-band** transitions:

**t_开 (heating half-band)**: from T_set - δ to T_set (heater ON):
```
┌────────────────────────────────────────────────────────────────┐
│ t_开 = (1/k) · ln((T_set - δ - T_ss) / (T_set - T_ss))       │
└────────────────────────────────────────────────────────────────┘
```

**t_关 (cooling half-band)**: from T_set + δ to T_set (heater OFF):
```
┌────────────────────────────────────────────────────────────────┐
│ t_关 = (1/k) · ln((T_set + δ - T_a) / (T_set - T_a))         │
└────────────────────────────────────────────────────────────────┘
```

### Zeno Effect

As δ → 0, both t_开 → ln(1) = 0 and t_关 → ln(1) = 0. The period P → 0, meaning **infinitely many switches in finite time**. This is the **Zeno phenomenon** from hybrid dynamical systems.

The half-band formulation directly demonstrates Zeno: the controller oscillates infinitely fast around T_set.

See: `Result/fig3_zeno_effect.png`

## 4. Step 2: Full-Band Resolution (T_L → T_H, T_H → T_L)

To resolve the Zeno problem, we use **full-band** transitions:

**t_开 (heating full-band)**: from T_L to T_H (heater ON):
```
┌────────────────────────────────────────────────────────────────┐
│ t_开 = (1/k) · ln((T_L - T_ss) / (T_H - T_ss))               │
└────────────────────────────────────────────────────────────────┘
```

**t_关 (cooling full-band)**: from T_H to T_L (heater OFF):
```
┌────────────────────────────────────────────────────────────────┐
│ t_关 = (1/k) · ln((T_H - T_a) / (T_L - T_a))                 │
└────────────────────────────────────────────────────────────────┘
```

### Derived Quantities

- **Period**: P = t_开 + t_关
- **Switches in T_total**: N = 2 · T_total / P
- **Heating energy per period**: E = U_max · t_开
- **Total energy**: E_总 = (N/2) · ε + (N/2) · E (where ε = switching cost per switch)

### Relationship: Full-Band ≈ 2 × Half-Band

The full-band switching times are approximately twice the half-band values (exact for small δ), since the temperature traverses 2δ instead of δ.

### Key Results (δ = 0.5 °C)

| Quantity | Value |
|----------|-------|
| T_H | 20.5 °C |
| T_L | 19.5 °C |
| t_开 (full-band) | 0.074 min |
| t_关 (full-band) | 0.667 min |
| Period P | 0.741 min |
| Switches (120 min) | 324 |

See: `Result/fig2_bangbang_oscillation.png`

### Comparison Across Delta Values

| δ (°C) | t_开 (min) | t_关 (min) | Period (min) | Switches/120min |
|---------|-----------|-----------|-------------|----------------|
| 0.1 | 0.015 | 0.133 | 0.148 | 1620 |
| 0.5 | 0.074 | 0.667 | 0.741 | 324 |
| 1.0 | 0.148 | 1.335 | 1.484 | 162 |
| 2.0 | 0.296 | 2.683 | 2.979 | 81 |
| 5.0 | 0.741 | 6.932 | 7.673 | 31 |

See: `Result/fig4_delta_comparison.png`

## 5. Energy and Optimal Delta Analysis

### Pure Heating Energy

The heating energy per period `E = U_max · t_开` is nearly constant across δ values (~180 for 120 min), because the average heating power required to maintain temperature near T_set against cooling is fixed by physics:

```
Average power = k · (T_set - T_a) = 0.1 × 15 = 1.5 °C/min
```

### Combined Cost (Energy + Comfort)

To find the optimal δ, we use a combined cost:

```
J(δ) = E_total(δ) + λ · RMSE²(δ) · T_total
```

Where:
- E_total = heating energy + switching energy (ε per switch)
- RMSE = root mean square deviation from T_set over one period
- λ = comfort weight

### Optimal Delta Results

| ε | λ | δ* | Period | Switches | RMSE |
|---|---|-----|--------|----------|------|
| 0.0 | any | ~0 | ~0 | → ∞ | ~0 |
| 1.0 | 1.0 | 1.27 | 1.89 | 127 | 0.74 |
| 1.0 | 10.0 | 0.59 | 0.88 | 274 | 0.34 |
| 5.0 | 1.0 | 2.17 | 3.24 | 74 | 1.26 |
| 5.0 | 10.0 | 1.01 | 1.50 | 160 | 0.58 |

Key insight: higher switching cost ε → larger optimal δ (fewer switches); higher comfort weight λ → smaller optimal δ (tighter temperature control).

See: `Result/fig5_energy_optimal_delta.png`

## 6. Full Simulation (Transient + Steady State)

Starting from T_initial = 10 °C with δ = 0.5 °C:
- Startup time to reach T_H = 20.5 °C: **0.752 min**
- Then enters periodic Bang-Bang oscillation

See: `Result/fig6_full_transient_simulation.png`

## 7. Summary of All Formulas

| Formula | Expression |
|---------|------------|
| General solution | T(t) = T_a + u/k + e^{-k(t-t0)} · (T(t0) - T_a - u/k) |
| t_开 (half-band) | (1/k) · ln((T_set - δ - T_ss) / (T_set - T_ss)) |
| t_关 (half-band) | (1/k) · ln((T_set + δ - T_a) / (T_set - T_a)) |
| t_开 (full-band) | (1/k) · ln((T_L - T_ss) / (T_H - T_ss)) |
| t_关 (full-band) | (1/k) · ln((T_H - T_a) / (T_L - T_a)) |
| Period | P = t_开 + t_关 |
| Switches | N = 2 · T_total / P |
| Energy per period | E = U_max · t_开 |
| Total energy | E_总 = (N/2) · ε + (N/2) · E |
| Combined cost | J = E_total + λ · RMSE² · T_total |

## 8. Files

| File | Description |
|------|-------------|
| `Week1_0D_BangBang_Analysis.ipynb` | Jupyter notebook with all code and analysis |
| `Result/fig1_heating_cooling_curves.png` | Heating and cooling analytical curves |
| `Result/fig2_bangbang_oscillation.png` | Bang-Bang oscillation with T_set, T_H, T_L, t_开, t_关 |
| `Result/fig3_zeno_effect.png` | Zeno effect: switching times vs δ |
| `Result/fig4_delta_comparison.png` | Oscillation comparison for multiple δ values |
| `Result/fig5_energy_optimal_delta.png` | Energy and optimal δ analysis |
| `Result/fig6_full_transient_simulation.png` | Full simulation with startup transient |
