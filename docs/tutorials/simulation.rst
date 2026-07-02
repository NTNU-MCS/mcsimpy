Vessel Simulations
==================

This tutorial shows the simplest use of the ``mcsimpy`` package. Here, we will show how you can use
the built in vessel simulation models found in ``mcsimpy.simulator``.


We start by defining the simulation parameter. We will use the maneuvering vessel simulation model of CSAD.
As this is a model scale vessel, we must use a small time step due to its high eigenfrequencies.

.. code-block:: python

    import numpy as np
    from mcsimpy.simulator import CSADMan3DOF

    dt = 0.01 # Simulation timestep corresponding to sampling frequency of 100 Hz
    simtime = 120   # 2 minutes simulation
    time = np.arange(0, simtime, dt)

    # Initialize a the simulation object
    vessel = CSADMan3DOF(dt=dt)

The vessel is integrated forward in time using 4th-order Runge-Kutta, the only supported integration method.

To have something more interesting than just a simulation without any loads, we will define a current and a constant load.

.. code-block:: python

    Uc = 0.01   # Current speed m/s
    beta_c = np.pi/4    # Current going north-west

    tau_const = np.array([1.0, 2.0, -1.3])  # Load in surge, sway, and yaw.

    # Start the simulation
    for i in range(1, len(time)):
        vessel.integrate(Uc, beta_c, tau_const)

