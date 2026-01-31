# Additional References for "Effective Control of Room Heating" Project

## Week 1: Newton's Law of Cooling, Bang-Bang Control, Feedback, and Hybrid Systems

These references supplement those already cited in the project brief (Wikipedia articles on Newton's law of cooling, the heat equation, and feedback; and Goebel, Sanfelice & Teel's 2009 IEEE CSM survey).

---

### Reference 1: ODE Foundations and Newton's Law of Cooling

**Citation:**
Boyce WE, DiPrima RC, Meade DB. *Elementary Differential Equations and Boundary Value Problems*. 12th ed. Hoboken, NJ: John Wiley & Sons; 2021. ISBN: 978-1-119-77769-4.

**Useful for:**
This is the standard undergraduate textbook for ordinary differential equations. Section 2.3 covers Newton's law of cooling in detail as an application of first-order linear ODEs, deriving the exponential decay solution T(t) = T_ambient + (T_0 - T_ambient) * exp(-kt). The textbook also includes thermostat-related heating/cooling cycle problems that are directly relevant to modelling room temperature dynamics. Essential for Week 1's mathematical foundation.

**Where to find:**
Available from university libraries and [Wiley](https://www.wiley.com). Also available on [Amazon](https://www.amazon.com/Elementary-Differential-Equations-Boundary-Problems/dp/1119777690).

---

### Reference 2: Nonlinear Dynamics and First-Order ODE Intuition

**Citation:**
Strogatz SH. *Nonlinear Dynamics and Chaos: With Applications to Physics, Biology, Chemistry, and Engineering*. 3rd ed. Boca Raton, FL: CRC Press; 2024. ISBN: 978-0-367-02650-9. DOI: [10.1201/9780429492563](https://doi.org/10.1201/9780429492563).

**Useful for:**
The opening chapters introduce first-order ODEs using geometric and graphical reasoning ("flows on the line"), with Newton's law of cooling as a motivating example. This geometric perspective is valuable for understanding how room temperature evolves as a dynamical system and for building intuition about stability and equilibrium before introducing control. Later chapters on bifurcations provide context for how system behaviour changes qualitatively with parameter variation.

**Where to find:**
[Publisher (CRC Press / Routledge)](https://www.routledge.com/Nonlinear-Dynamics-and-Chaos-With-Applications-to-Physics-Biology-Chemistry-and-Engineering/Strogatz/p/book/9780367026509). Widely available in university libraries.

---

### Reference 3: Feedback Systems and PID Control

**Citation:**
Astrom KJ, Murray RM. *Feedback Systems: An Introduction for Scientists and Engineers*. 2nd ed. Princeton, NJ: Princeton University Press; 2021. ISBN: 978-0-691-19398-4.

**Useful for:**
This is an outstanding introductory textbook covering feedback control from first principles. It includes state-space modelling, stability analysis via Lyapunov functions, transfer functions, Nyquist analysis, and a thorough treatment of PID control -- all directly applicable to designing a room heating controller. The book is freely available online, making it highly accessible for group projects. Chapter 11 on PID control is particularly relevant for comparing bang-bang and proportional-integral-derivative approaches to thermostat design.

**Where to find:**
Full text freely available at [https://www.cds.caltech.edu/~murray/amwiki/](https://fbswiki.org/wiki/index.php/Feedback_Systems:_An_Introduction_for_Scientists_and_Engineers) and [Caltech PDF](https://www.cds.caltech.edu/~murray/books/AM08/pdf/am08-complete_28Sep12.pdf). Published by [Princeton University Press](https://press.princeton.edu/books/hardcover/9780691193984/feedback-systems).

---

### Reference 4: Hybrid Dynamical Systems (Comprehensive Textbook)

**Citation:**
Goebel R, Sanfelice RG, Teel AR. *Hybrid Dynamical Systems: Modeling, Stability, and Robustness*. Princeton, NJ: Princeton University Press; 2012. ISBN: 978-0-691-15389-6. DOI: [10.1515/9781400842636](https://doi.org/10.1515/9781400842636).

**Useful for:**
This is the definitive textbook by the same authors as the 2009 IEEE survey article already cited in the project brief. It provides a complete mathematical framework for hybrid systems that combine continuous flows and discrete jumps -- exactly the structure of a thermostat switching a heater on and off while room temperature evolves continuously. The book covers modelling, stability theory (Lyapunov methods for hybrid systems), and robustness. It includes the thermostat as a running example throughout. Essential for formalising the bang-bang thermostat as a hybrid automaton.

**Where to find:**
[Princeton University Press](https://press.princeton.edu/books/hardcover/9780691153896/hybrid-dynamical-systems). Companion materials at [https://hybrid.soe.ucsc.edu/hsbook](https://hybrid.soe.ucsc.edu/hsbook).

---

### Reference 5: Zeno Behaviour in Hybrid Systems

**Citation:**
Zhang J, Johansson KH, Lygeros J, Sastry SS. Zeno hybrid systems. *International Journal of Robust and Nonlinear Control*. 2001;11(5):435-451. DOI: [10.1002/rnc.592](https://doi.org/10.1002/rnc.592).

**Useful for:**
This foundational paper defines and analyses the Zeno phenomenon in hybrid systems -- the occurrence of infinitely many discrete transitions in finite time. While a physical thermostat cannot switch infinitely fast, idealised models of bang-bang controllers with zero hysteresis can exhibit Zeno behaviour. Understanding when and why Zeno executions arise (and how to regularise them via hysteresis/deadband) is critical for ensuring that thermostat models are well-posed and simulatable. The paper gives necessary and sufficient conditions for Zeno executions and introduces the concept of the Zeno set.

**Where to find:**
[Wiley Online Library](https://onlinelibrary.wiley.com/doi/abs/10.1002/rnc.592).

---

### Reference 6: Dynamical Properties of Hybrid Automata

**Citation:**
Lygeros J, Johansson KH, Simic SN, Zhang J, Sastry SS. Dynamical properties of hybrid automata. *IEEE Transactions on Automatic Control*. 2003;48(1):2-17. DOI: [10.1109/TAC.2002.806650](https://doi.org/10.1109/TAC.2002.806650).

**Useful for:**
This paper studies hybrid automata from a dynamical systems perspective, establishing fundamental properties such as existence and uniqueness of executions, continuity of solutions with respect to initial conditions, and conditions for non-blocking behaviour. It provides the rigorous mathematical framework needed to prove that a thermostat hybrid automaton model is well-defined. The paper bridges control theory and computer science perspectives on hybrid systems, which is useful for understanding the thermostat as both a control system and a state machine.

**Where to find:**
[IEEE Xplore](https://ieeexplore.ieee.org/document/1166520/).

---

### Reference 7: Numerical Methods for the Heat Equation

**Citation:**
Strikwerda JC. *Finite Difference Schemes and Partial Differential Equations*. 2nd ed. Philadelphia, PA: SIAM; 2004. ISBN: 978-0-898716-39-9.

**Useful for:**
This textbook provides a rigorous treatment of finite difference methods for PDEs, with the heat equation as the primary example throughout. It covers the explicit (FTCS), implicit (BTCS), and Crank-Nicolson schemes, along with stability analysis (von Neumann method), consistency, and convergence. Directly applicable if the project extends the lumped-parameter (ODE) model to a spatially distributed (PDE) model of room temperature. The stability condition r <= 1/2 for the explicit scheme and the unconditional stability of implicit methods are key practical considerations for simulation.

**Where to find:**
[SIAM Bookstore](https://epubs.siam.org/doi/book/10.1137/1.9780898717938). Available in university libraries.

---

### Reference 8: PID Control for HVAC Systems

**Citation:**
Blasco C, Monreal J, Benitez I, Lluna A. Modelling and PID control of HVAC system according to energy efficiency and comfort criteria. In: Perez JB, et al., editors. *Trends in Practical Applications of Agents and Multiagent Systems*. Advances in Intelligent and Soft Computing, vol. 157. Berlin: Springer; 2012. p. 365-374. DOI: [10.1007/978-3-642-27509-8_31](https://doi.org/10.1007/978-3-642-27509-8_31).

**Useful for:**
This paper applies PID control specifically to HVAC systems in office buildings, optimising for both energy efficiency and thermal comfort. It demonstrates how to model the thermal dynamics of a building zone and then design PID controllers for temperature regulation. Provides a practical, applied contrast to the theoretical bang-bang approach: PID controllers modulate heating power continuously rather than switching fully on/off, typically achieving better steady-state accuracy and energy efficiency. Useful for the project's comparison of control strategies.

**Where to find:**
[Springer Link](https://link.springer.com/chapter/10.1007/978-3-642-27509-8_31).

---

## Week 2+: LQR Optimal Control, Pontryagin Minimum Principle

---

### Reference 9: Linear Quadratic Optimal Control (LQR)

**Citation:**
Anderson BDO, Moore JB. *Optimal Control: Linear Quadratic Methods*. Englewood Cliffs, NJ: Prentice-Hall; 1990. ISBN: 978-0-13-638560-0. (Reprinted by Dover, 2007. ISBN: 978-0-486-45766-6.)

**Useful for:**
This is the definitive textbook on Linear Quadratic Regulator (LQR) theory. It covers the quadratic cost functional J = ∫[x'Qx + u'Ru]dt, the derivation of the algebraic Riccati equation (ARE), existence and uniqueness of solutions, and the resulting optimal state-feedback gain K = R⁻¹B'P. The book provides both continuous-time and discrete-time formulations. For our project, the scalar case (single-state room temperature model) simplifies the ARE to a quadratic equation with an analytical solution, making it ideal for demonstrating optimal control theory in an accessible setting. Chapter 1 (introduction to LQR) and Chapter 4 (the ARE) are most relevant.

**Where to find:**
[Dover Publications](https://store.doverpublications.com/products/9780486457666). Available in university libraries. The Dover reprint is inexpensive (~$20).

---

### Reference 10: Calculus of Variations and Optimal Control (Pontryagin)

**Citation:**
Liberzon D. *Calculus of Variations and Optimal Control Theory: A Concise Introduction*. Princeton, NJ: Princeton University Press; 2012. ISBN: 978-0-691-15187-8.

**Useful for:**
This concise textbook provides a rigorous yet accessible treatment of the Pontryagin minimum principle. Chapters 4-5 cover the formulation of optimal control problems with state constraints, the Hamiltonian, costate (adjoint) equations, transversality conditions, and the resulting two-point boundary value problem (TPBVP). The thermostat control problem — minimising a quadratic cost subject to ODE dynamics and control bounds [0, U_max] — is a canonical example of constrained optimal control. The book's clear exposition of necessary conditions for optimality (including the switching structure when control bounds are active) directly supports our Pontryagin controller implementation.

**Where to find:**
[Princeton University Press](https://press.princeton.edu/books/hardcover/9780691151878/calculus-of-variations-and-optimal-control-theory). A pre-publication version is freely available from the author's website: [https://liberzon.csl.illinois.edu/teaching/cvoc/](https://liberzon.csl.illinois.edu/teaching/cvoc/).

---

### Reference 11: Optimal Control Theory — Introduction (Pontryagin)

**Citation:**
Kirk DE. *Optimal Control Theory: An Introduction*. Mineola, NY: Dover; 2004. ISBN: 978-0-486-43484-1. (Originally published by Prentice-Hall, 1970.)

**Useful for:**
This classic textbook provides a thorough introduction to optimal control theory with emphasis on the minimum principle. Chapter 5 derives the Pontryagin minimum principle from first principles, while Chapter 6 covers constrained optimal control problems (control magnitude constraints, as in our bounded heater power). The book includes numerous worked examples of forming the Hamiltonian, deriving costate equations, and solving the resulting TPBVP — exactly the methodology used in our `pontryagin.py` controller. The Dover reprint makes this seminal text widely accessible.

**Where to find:**
[Dover Publications](https://store.doverpublications.com/products/9780486434841). Also available on [Amazon](https://www.amazon.com/Optimal-Control-Theory-Introduction-Engineering/dp/0486434842). University libraries typically hold copies.

---

## Summary Table

| # | Reference | Topic Area | Type | Used in |
|---|-----------|-----------|------|---------|
| 1 | Boyce, DiPrima & Meade (2021) | Newton's law of cooling, ODEs | Textbook | ODE model |
| 2 | Strogatz (2024) | Nonlinear dynamics, geometric ODE analysis | Textbook | ODE model |
| 3 | Astrom & Murray (2021) | Feedback control, PID control, LQR | Textbook (free) | PID, LQR |
| 4 | Goebel, Sanfelice & Teel (2012) | Hybrid dynamical systems | Textbook | Bang-Bang |
| 5 | Zhang et al. (2001) | Zeno behaviour in hybrid systems | Journal paper | Bang-Bang |
| 6 | Lygeros et al. (2003) | Hybrid automata properties | Journal paper | Bang-Bang |
| 7 | Strikwerda (2004) | Finite difference methods, heat equation | Textbook | 1D/2D PDE |
| 8 | Blasco et al. (2012) | PID control for HVAC | Conference paper | PID |
| 9 | Anderson & Moore (1990/2007) | Linear Quadratic Regulator (LQR) | Textbook | LQR |
| 10 | Liberzon (2012) | Calculus of variations, Pontryagin principle | Textbook (free pre-print) | Pontryagin |
| 11 | Kirk (2004) | Optimal control theory, minimum principle | Textbook | Pontryagin |
