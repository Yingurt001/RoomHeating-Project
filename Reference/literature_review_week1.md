# Week 1 Literature Review Summary

## Objective

Review foundational literature on Newton's law of cooling, feedback control, bang-bang thermostats, and hybrid dynamical systems to establish the mathematical framework for modelling automated room heating.

---

## 1. Newton's Law of Cooling

Newton's law of cooling models heat loss as proportional to the temperature difference between an object and its surroundings: $dT/dt = -k(T - T_a)$. This first-order linear ODE has the analytical solution $T(t) = T_a + (T_0 - T_a)e^{-kt}$, showing exponential decay toward ambient temperature with time constant $\tau = 1/k$ (Boyce & DiPrima, 2021, §2.3). The constant $k$ encapsulates insulation quality, wall surface area, and room volume. Adding a heater input $u(t)$ yields $dT/dt = -k(T - T_a) + u(t)$, which shifts the steady-state to $T_{ss} = T_a + U_{max}/k$ when the heater runs continuously. Strogatz (2024) provides geometric intuition for this as a "flow on the line" — the fixed point $T_{ss}$ is stable because the slope of $dT/dt$ vs $T$ is $-k < 0$.

## 2. Feedback Control and Bang-Bang Thermostats

Real thermostats operate as feedback controllers: they measure the room temperature and decide whether to turn the heater on or off (Astrom & Murray, 2021). The simplest strategy is **bang-bang control** — a binary ON/OFF switch at a set-point $T_{set}$. This creates a piecewise-defined ODE: the system follows one set of dynamics (heating) when $T < T_{set}$ and another (cooling) when $T \geq T_{set}$.

The key practical issue is **chattering**: without a dead band, the heater switches infinitely fast near $T_{set}$. Real thermostats use **hysteresis** — a dead band $[T_{set} - \delta, T_{set} + \delta]$ — to ensure a minimum ON/OFF duration. The trade-off is clear: larger $\delta$ means fewer switches but greater temperature fluctuation around the set-point. Our numerical experiments confirm that the oscillation period scales roughly linearly with $\delta$ (from 0.37 min at $\delta = 0.25°C$ to 7.67 min at $\delta = 5°C$).

## 3. Hybrid Dynamical Systems and the Zeno Effect

The thermostat-heater system is a **hybrid dynamical system**: it combines continuous evolution (temperature governed by an ODE) with discrete transitions (heater switching). Goebel, Sanfelice & Teel (2012) provide the definitive mathematical framework, modelling such systems as inclusions with a flow set and a jump set. Their book uses the thermostat as a running example.

A critical theoretical concern is the **Zeno phenomenon**: infinitely many discrete transitions occurring in finite time. Zhang et al. (2001) establish formal conditions for when Zeno behaviour arises in hybrid systems. For the bang-bang thermostat without hysteresis ($\delta = 0$), our simulations demonstrate that the system reaches $T_{set}$ and then undergoes 500 switches in less than 0.01 minutes — a numerical manifestation of Zeno behaviour. Introducing hysteresis ($\delta > 0$) eliminates this pathology by guaranteeing a minimum time between switches. Lygeros et al. (2003) provide additional rigour on existence and uniqueness of executions for such hybrid automata.

## 4. Numerical Methods (Preview for Later Phases)

When the model is extended to include spatial variation (the heat equation $\partial T/\partial t = \alpha \nabla^2 T + u$), finite difference methods become necessary. Strikwerda (2004) covers the standard schemes: explicit FTCS (conditionally stable, $\Delta t \leq \Delta x^2 / 2\alpha$), implicit BTCS (unconditionally stable), and Crank-Nicolson (second-order accurate in time). For HVAC-specific PID control as an alternative to bang-bang, Blasco et al. (2012) demonstrate improved energy efficiency in building simulations.

---

## Key Takeaways for the Project

1. The ODE model $dT/dt = -k(T - T_a) + u(T)$ is well-understood analytically — we can derive exact formulas for the oscillation period and duty cycle.
2. Bang-bang control without hysteresis leads to Zeno behaviour (infinite switching); hysteresis is both physically realistic and mathematically necessary.
3. The trade-off between comfort (small $\delta$, temperature close to $T_{set}$) and efficiency (fewer switches, longer equipment life) is a central design question.
4. Extending to PDE models will require careful numerical scheme selection — Crank-Nicolson is the recommended starting point for balancing accuracy and stability.

---

## References

1. Astrom KJ, Murray RM. *Feedback Systems*. 2nd ed. Princeton University Press; 2021.
2. Blasco C et al. Modelling and PID control of HVAC system. *Trends in Practical Applications of Agents*. Springer; 2012. p. 365-374.
3. Boyce WE, DiPrima RC, Meade DB. *Elementary Differential Equations*. 12th ed. Wiley; 2021.
4. Goebel R, Sanfelice RG, Teel AR. *Hybrid Dynamical Systems*. Princeton University Press; 2012.
5. Lygeros J et al. Dynamical properties of hybrid automata. *IEEE Trans. Automatic Control*. 2003;48(1):2-17.
6. Strikwerda JC. *Finite Difference Schemes and PDEs*. 2nd ed. SIAM; 2004.
7. Strogatz SH. *Nonlinear Dynamics and Chaos*. 3rd ed. CRC Press; 2024.
8. Zhang J et al. Zeno hybrid systems. *Int. J. Robust Nonlinear Control*. 2001;11(5):435-451.
