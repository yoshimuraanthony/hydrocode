# 2D Rayleigh-Taylor hydrocode 

Simulate time evolution of field variables rho, p, and **v** in inviscid fluid
subject to Rayleigh Taylor (RT) instability.  Algorithm based on lecture 16.

## Boundary conditions:

* z-direction: hard boundaries a distance *d* above and below the interface
* x-direction: periodic or reflective boundary conditions

## Usage

```
# initialize fluid class
f = Fluid()

# time evolve the fluid
for n in range(Nt):
    f.update()

# get field variable values
rho_a2 = f.getRho()
p_a2 = f.getP()
v_a3 = f.getV()

# plot field variable values as heatmaps
f.plot('rho')
f.plot('p')
f.plot('vx')
f.plot('vz')

# plot animation of field variable evolution
f.saveAnimation('rho')
```

## To do (in descending priority)

1. Write the update function
    1. write enforceBCs (Sara)
    2. write predictor steps (Ali and Pete *and Kory*)
        0. note: use corrected dt evaluated at t (from previous corrector step)
        1. get dvar/dt with forward spacial derivative at t (eq. 16)
        2. get predicted variables "varp" at t+dt (eq. 17)
        3. enforce BCs on predicted variables
    3. write corrector steps
        1. get dvarp/dt with backward spacial der at t+dt (eq. 18)
        2. get dvar/dt(avg) at t+dt/2 (eq. 19)
        3. get corrected variables at t+dt (equation between 19 and 20)
        4. enforce BCs on corrected variables
        5. get dt evaluated at t+dt using corrected variables (run getDt)
    4. check for conservation (optional)
        1. Mass (self.rho_a2.sum())
        2. Momentum
        3. Energy
2. Testing script (*Kory*)
3. Create a movie plot
    1. show time in corner
    2. take scalar field variable as argument
    3. plot dumps instead of every frame
4. (Optional) Enable writing/reading of restart files 
