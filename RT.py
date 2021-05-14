from numpy import array, arange, zeros, ones, linspace, newaxis, where
from numpy import exp, log, cosh, sinh, tanh, cos, sin, tan, sqrt, ceil
from numpy import pi
from numpy import roll  # in place of update function
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable

# catch warnings (e.g. RuntimeWarning) as an exception
import warnings
warnings.filterwarnings('error')

"""Time evolve 2D field variables in classical 2D RT system.

Algorithm based on lecture 16

The Fluid class holds all of the field variables and dictates their evolution.
The class structure eliminates the need to pass a ton of arguments into every
function, though it does require that "self" be written in front of all the
class variables, which can be annoying.

Notation:
    Variables named <var>_a<n> are arrays of rank n.
    Arrays are indexed by [vector component, x grid point, z grid point].

    p_a2[50,60] is the pressure at grid point (50, 60)
    v_a3[0,20,30] is the x-component of the velocity at grid point (20, 30)
    F_a4[1,0,5,5] is the (z,x)-element of the F tensor at grid point (5, 5)

Docstrings loosely follow the Google Python Style Guide
https://google.github.io/styleguide/pyguide.html
"""

class Fluid(object):
    """Discretized classical 2D RT fluid with periodic boundary conditions

    Attributes:
        rho, p: scalar field variables
        v: vector field variable
    """
    def __init__(self,
            gamma=5./3., g=1.,  # m/s**2
            xmax=.05, d=.5, Nx=50, Nz=500,  # dimensions and resolution
            rhoh=1., rhol=.5, L0=.02,  # initial density profile parameters
            pd=1.,  # Pa, initial pressure parameters
            vpert=.01,  # m/s
            tdump=.1, ttot=1.85, tscale=1,  # s
            ):
        """Initialize equilibrium 2D field variables.

        gamma: cp/cv (pos float)
        g: magnitude of gravity in m/s^2 (pos float)
        {xmax,d}: distance from origin to {x,z} boundary in m (pos int)
        N{x,z}: number of grid points in {x,z} direction (pos int)
        rho{h,l}: heavy and light densities in kg/m^3 (pos float)
        L0: "width" of interface in m (pos float)
        pd: pressure at upper z boundary z=d in Pa (float)
        vpert: velocity perturbation amplitude in m/s (float)
        tdump: time between dumps in s (pos float)
        ttot: total simulation time in s (pos float)
        tscale: scaling of time step (pos float)
        """
        # physical constants
        self.g = g  # m/s**2
        self.gamma = gamma

        # domain size and resolution
        self.xmax = xmax  # m
        self.d = d  # m
        self.Nx = Nx
        self.Nz = Nz
        self.dx = 2*xmax/Nx  # m
        self.dz = 2*d/Nz  # m
        self.dmin = min((self.dx, self.dz))

        # relating to initial density
        self.rhoh = rhoh  # kg/m**3
        self.rhol = rhol  # kg/m**3
        self.L0 = L0  # m
        
        # relating to initial pressure
        self.pd = pd  # Pa

        # relating to velocity
        self.vpert = vpert  # Pa

        # x and z coordinates
        self.x_a1 = linspace(-xmax, xmax, Nx)  # m
        self.z_a1 = linspace(-d, d, Nz)  # m 

        # equilibrium density and pressure vs z
        self.setRho0() 
        self.setP0()

        # initial primitive field variables: rho, p, and v
        self.rho_a2 = self.rho0_a1 + zeros((Nx, Nz))
        self.p_a2 = self.p0_a1 + zeros((Nx, Nz))
        self.v_a3 = zeros((2, Nx, Nz))  # 2D vector field
        self.perturb()
        
        # initial conservative field variables
        self.G_a3 = self._getG(self.rho_a2)
        self.W_a2 = self._getW(self.rho_a2)
        self.G_a3, self.rho_a2, self.W_a2 = self._enforceBCs(
                self.G_a3, self.rho_a2, self.W_a2)

        self.F_a4 = self._getF(self.rho_a2)
        self.Q_a3 = self._getQ(self.rho_a2)
        self.SG_a3 = self._getSG(self.rho_a2)
        self.SW_a2 = self._getSW(self.rho_a2)

        # time step and time elapsed
        self.tscale = tscale
        self.ttot = ttot
        self.tdump = tdump
        self.elapsed = 0.
        self.nextdump = 0.
        self.nsteps = 0
        self.ndumps = 0
        self._setdt()

        # animation frames
        self.rho_list = []
        self.p_list = []
        self.vx_list = []
        self.vz_list = []
        self.G_list = []
        self.W_list = []

        self.mass_list = []
        self.tdump_list = []
        self.dtdump_list = []
        self.omega_list = []

        # conservation tests
        self.Mass_i = self.getMass()
        self.Energy_i = self.getEnergy()

        # history of quantities
        self.mass_list = [self.Mass_i]
        self.energy_list = [self.Energy_i]
        
        # corrector step needs previous and predicted rho, G, and W
        self.prho_a2 = self.rho_a2.copy() 
        self.pG_a3 = self.G_a3.copy()
        self.pW_a2 = self.W_a2.copy()

    # ---------------------- initialize primitvies -----------------------
    def setRho0(self):
        """Set equilibrium rho0(z) (lec 16 sl 4)."""
        self.rho0_a1 = (self.rhoh - self.rhol) \
                * (tanh(self.z_a1/self.L0) + 1) \
                / 2 + self.rhol

    def setP0(self):
        """Set equilibrium p0(z) = p0(d) + Int(rho0, z) (lec 16 sl 4)."""
        self.p0_a1 = self.pd + self.g / 2 * ( \
                (self.rhoh-self.rhol) * self.L0 \
                * log((cosh(self.d/self.L0) / cosh(self.z_a1/self.L0))) \
                + (self.rhoh+self.rhol) * (self.d-self.z_a1)
                )

    def perturb(self):
        """Perturb the field variables from equilibrium."""
        k = pi/self.xmax/2
        x_a2 = self.x_a1[:, newaxis] + self.xmax
        z_a2 = self.z_a1[newaxis, :]
        self.v_a3[0] = self.vpert*sin(k*x_a2) \
                * exp(-k*abs(tanh(z_a2/self.L0)*z_a2)) \
                * tanh(z_a2/self.L0)
        self.v_a3[1] = self.vpert*cos(k*x_a2) \
                * exp(-k*abs(tanh(z_a2/self.L0)*z_a2)) \

    # -------------------------- plot heatmaps ---------------------------
    def plotInit(self, var='rho'):
        """Plot equilibrium density or pressure vs z.

        var: variable to be plotted, e.g. rho or p (str)
        """
        if var[0]=='r':
            var_a1 = self.rho0_a1
            label = r"$\rho$ (kg/m$^3$)"
        elif var[0]=='p':
            var_a1 = self.p0_a1
            label = "p (Pa)"

        fig, ax = plt.subplots()
        ax.plot(self.z_a1, var_a1, lw=2)
        ax.set_xlabel('z (m)', fontsize=16)
        ax.set_ylabel(label, fontsize=16)
        plt.tight_layout()
        plt.show()

    def plot(self, var='rho', cmap='plasma'):
        """Plot scalar field variable at current timestep as 2D heatmap.

        var: variable to be plotted, e.g. p, rho, vz, or vx (str)
        cmap: color scheme for heatmap (str)
            (https://matplotlib.org/stable/tutorials/colors/colormaps.html)
        """
        if var[0]=='r' or var[0]=='R':
            var_a2 = self.rho_a2
            label = r"$\rho$ (kg/m$^3$)"
        elif var[0]=='p' or var[0]=='P':
            var_a2 = self.p_a2
            label = "$p$ (Pa)"
        elif 'v' in var and 'x' in var:
            var_a2 = self.v_a3[0]
            label = "$v_x$ (m/s)"
        elif 'v' in var and 'z' in var:
            var_a2 = self.v_a3[1]
            label = "$v_z$ (m/s)"
        else:
            print('unrecognized variable')
            return

        # imshow plots elements in reverse with first index on vertical axis
        var_a2 = var_a2.T[::-1,::-1]

        # heatmap
        fig, ax = plt.subplots(figsize=(3.5,8))
        im = ax.imshow(var_a2,
                extent = [-self.xmax, self.xmax, -self.d, self.d],
                cmap=cmap, interpolation='nearest')
        ax.set_xlabel('x (m)', fontsize=16)
        ax.set_ylabel('z (m)', fontsize=16)

        # color bar on right side
        cb = fig.colorbar(im)
        cb.set_label(label=label, fontsize=16)

        plt.tight_layout()
        plt.show()

    # -------------------- get copies of primitives ----------------------
    def getRho(self):
        """Return copy of rho at current time step."""
        return self.rho_a2.copy()

    def getP(self):
        """Return copy of p at current time step."""
        return self.p_a2.copy()

    def getVx(self):
        """Return copy of vx at current time step."""
        return self.v_a3.copy()[0]

    def getVz(self):
        """Return copy of vz at current time step."""
        return self.v_a3.copy()[1]

    # ----------------------- get field variables -------------------------
    def _getv(self, G_a3, rho_a2):
        """Set velocity from conservatives (typed eq 11)."""
        return G_a3 / rho_a2

    def _getp(self, G_a3, rho_a2, W_a2):
        """Set pressure from conservatives (typed eq 11)."""
        return (self.gamma - 1) * (W_a2 - (G_a3**2).sum(0)/rho_a2/2)

    def _getG(self, rho_a2):
        """Set momentum from primitives (lec 16 sl 14)."""
        return rho_a2 * self.v_a3

    def _getF(self, rho_a2):
        """Set momentum flux from primitives (lec 16 sl 14)."""
        F_a4 =  rho_a2[newaxis, newaxis, :, :] \
                * self.v_a3[newaxis, :, :, :] \
                * self.v_a3[:, newaxis, :, :]
        F_a4[0, 0] += self.p_a2
        F_a4[1, 1] += self.p_a2
        return F_a4

    def _getW(self, rho_a2):
        """Set energy from primitives (lec 16 sl 14)."""
        return self.p_a2 / (self.gamma-1) \
                + rho_a2 * (self.v_a3**2).sum(0) / 2

    def _getQ(self, rho_a2):
        """Set energy flux from primitives (lec 16 sl 14)."""
        return self.v_a3 \
                * (self.gamma*self.p_a2 / (self.gamma-1) \
                + rho_a2 * (self.v_a3**2).sum(0) / 2)

    def _getSG(self, rho_a2):
        """Set momentum source from primitives (lec 16 sl 17)."""
        SG_a3 = zeros((2, self.Nx, self.Nz))
        SG_a3[1] = -self.g * rho_a2  # g is nonzero only in z-direction
        return SG_a3

    def _getSW(self, rho_a2):
        """Set energy source from primitives (lec 16 sl 17)."""
        return -self.g * rho_a2 * self.v_a3[1]

    # ------------------ enforce boundary conditions ---------------------
    def _enforceBCs(self, G_a3, rho_a2, W_a2):
        """Enforce reflective boundary conditions.

        Act on rho, G, W, and pi after they are updated
        """
        # Define some commonly used variables
        g  = self.g
        dz = self.dz
        gamma = self.gamma

        # Apply Reflective Boundary Conditions in the x-direction
        # i = 0 BCs
        G_a3[0,0,:] = -G_a3[0,1,:]
        G_a3[1,0,:] = G_a3[1,1,:]
        W_a2[0,:] = W_a2[1,:]
        rho_a2[0,:] = rho_a2[1,:]

        # i = Nx+1 BCs
        G_a3[0,-1,:] = -G_a3[0,-2,:]
        G_a3[1,-1,:] = G_a3[1,-2,:]
        W_a2[-1,:] = W_a2[-2,:]
        rho_a2[-1,:] = rho_a2[-2,:]

        # Apply Rigid Wall Boundary Conditions in the z-direction
        # j = 0 BCs
        G_a3[0,:,0] = G_a3[0,:,1]
        G_a3[1,:,0] = -G_a3[1,:,1]
        rho_a2[:,0] = rho_a2[:,1]

        # j = Nz+1 BCs
        G_a3[0,:,-1] = G_a3[0,:,-2]
        G_a3[1,:,-1] = -G_a3[1,:,-2]
        rho_a2[:,-1] = rho_a2[:,-2]

        # initialize pi
        p_a2 = (gamma - 1) * (W_a2 - (G_a3**2).sum(0)/rho_a2/2)
        pi_a2 = p_a2 + G_a3[1]**2/rho_a2
        pi_a2[:,0] = pi_a2[:,1] + rho_a2[:,1]*g*dz
        pi_a2[:,-1] = pi_a2[:,-2] - rho_a2[:,-2]*g*dz

        # w(i,0)  =  ... ###
        Gx2 = G_a3[0,:,0]**2
        Gz2 = G_a3[1,:,0]**2
        term1 = pi_a2[:,0] - Gz2/rho_a2[:,0]
        term2 = Gx2 + Gz2 
        W_a2[:,0] = term1/(gamma - 1.) + term2/2./rho_a2[:,0]

        # w(i,-1)  =  ... ###
        Gx2 = G_a3[0,:,-1]**2
        Gz2 = G_a3[1,:,-1]**2
        term1 = pi_a2[:,-1] - Gz2/rho_a2[:,-1]
        term2 = Gx2 + Gz2 
        W_a2[:,-1] = term1/(gamma - 1.) + term2/2./rho_a2[:,-1]

        return G_a3, rho_a2, W_a2
            
    # ----------------------- check conservation -------------------------
    def getMass(self):
        """Return total mass in kg."""
        return self.rho_a2.sum()

    def getPE(self):
        """Total potential energy"""
        PE = (self.rho_a2*self.z_a1*self.g).sum()
        return PE
    
    def getKE(self):
        """Total kinetic energy"""
        KE = self.W_a2.sum()
        return KE
    
    def getEnergy(self):
        """Return total energy"""        
        return self.getPE() + self.getKE()

    def getOmega(self):
        """Return vorticity magnitude"""
        dvzdx_a2 = (self.v_a3[1,1:,:-1] - self.v_a3[1,:-1,:-1]) / self.dx
        dvxdz_a2 = (self.v_a3[0,:-1,1:] - self.v_a3[0,:-1,:-1]) / self.dz
        return abs(dvzdx_a2 - dvxdz_a2).sum()

    def checkConservation(self, tol=1e-5):
        """Check for Conservation of mass, energy, and momentum."""
        Mass_f = self.getMass()
        Energy_f = self.getEnergy()
        Momentum_f = self.G_a3[0].sum()

        deltaM = (Mass_f - self.Mass_i) / self.Mass_i
        deltaE = (Energy_f - self.Energy_i) / self.Energy_i
        
        if abs(deltaM) < tol:
            print('  Total Mass is Conserved')
        else:
            print('  Total Mass not Conserved by {:.3g}%'.format(100*deltaM))
        
        if abs(deltaE) < tol:
            print('  Total Energy is Conserved')
        else:
            print('  Total Energy not Conserved by {:.3g}%'.format(100*deltaE))
        
        self.mass_list.append(Mass_f)
        self.energy_list.append(Energy_f)

    # -------------------- advance field variables -----------------------
    def _setdt(self):
        """Set time step in seconds."""
        cs_a2 = sqrt(self.gamma*self.p_a2/self.rho_a2)
        vmag_a2 = sqrt((self.v_a3**2).sum(0))
        self.dt = self.dmin / (cs_a2 + vmag_a2).max() / 2 * self.tscale

    def _setPredictedVars(self):
        """Set "predictor" field variables.

        Update using primitives at current timestep (lect 16 sl 14).
        """
        # advance conservatives with forward differences
        self.prho_a2[:-1,:-1] = self.rho_a2[:-1,:-1] \
            + self.dt * ( \
                - (self.G_a3[0,1:,:-1] - self.G_a3[0,:-1,:-1]) / self.dx \
                - (self.G_a3[1,:-1,1:] - self.G_a3[1,:-1,:-1]) / self.dz \
                + 0
            )
    
        self.pG_a3[:,:-1,:-1] = self.G_a3[:,:-1,:-1] \
            + self.dt * ( \
                - (self.F_a4[0,:,1:,:-1] - self.F_a4[0,:,:-1,:-1]) / self.dx \
                - (self.F_a4[1,:,:-1,1:] - self.F_a4[1,:,:-1,:-1]) / self.dz \
                + self.SG_a3[:,:-1,:-1]
            )

        self.pW_a2[:-1,:-1] = self.W_a2[:-1,:-1] \
            + self.dt * ( \
                - (self.Q_a3[0,1:,:-1] - self.Q_a3[0,:-1,:-1]) / self.dx \
                - (self.Q_a3[1,:-1,1:] - self.Q_a3[1,:-1,:-1]) / self.dz \
                + self.SW_a2[:-1,:-1]
            )

        # enforce boundary conditions
        self.pG_a3, self.prho_a2, self.pW_a2 = self._enforceBCs(
                self.pG_a3, self.prho_a2, self.pW_a2)

        # advance remaining primitives
        self.v_a3 = self._getv(self.pG_a3, self.prho_a2)
        self.p_a2 = self._getp(self.pG_a3, self.prho_a2, self.pW_a2)

        # advance remaining conservatives
        self.F_a4 = self._getF(self.prho_a2)
        self.Q_a3 = self._getQ(self.prho_a2)
        self.SG_a3 = self._getSG(self.prho_a2)
        self.SW_a2 = self._getSW(self.prho_a2)

    def _setCorrectedVars(self):
        """Set "corrector" variables.

        Update field vars with avg time derivatives (lect 16 sl 15).
        """
        # advance conservatives with forward differences
        self.rho_a2[1:,1:] = 0.5 * (self.rho_a2[1:,1:] + self.prho_a2[1:,1:]) \
            + (self.dt / 2.0) * ( \
                - (self.pG_a3[0,1:,1:] - self.pG_a3[0,:-1,1:]) / self.dx \
                - (self.pG_a3[1,1:,1:] - self.pG_a3[1,1:,:-1]) / self.dz \
                + 0
            )
        
        self.G_a3[:,1:,1:] = 0.5 * (self.G_a3[:,1:,1:] + self.pG_a3[:,1:,1:]) \
            + (self.dt / 2.0) * ( \
                - (self.F_a4[0,:,1:,1:] - self.F_a4[0,:,:-1,1:]) / self.dx \
                - (self.F_a4[1,:,1:,1:] - self.F_a4[1,:,1:,:-1]) / self.dz \
                + self.SG_a3[:,1:,1:]
            )

        self.W_a2[1:,1:] = 0.5 * (self.W_a2[1:,1:] + self.pW_a2[1:,1:]) \
            + (self.dt / 2.0) * ( \
                - (self.Q_a3[0,1:,1:] - self.Q_a3[0,:-1,1:]) / self.dx \
                - (self.Q_a3[1,1:,1:] - self.Q_a3[1,1:,:-1]) / self.dz \
                + self.SW_a2[1:,1:]
            )

        # enforce boundary conditions
        self.G_a3, self.rho_a2, self.W_a2 = self._enforceBCs(
                self.G_a3, self.rho_a2, self.W_a2)

        # advance remaining primitives
        self.v_a3 = self._getv(self.G_a3, self.rho_a2)
        self.p_a2 = self._getp(self.G_a3, self.rho_a2, self.W_a2)

        # advance remaining conservatives
        self.F_a4 = self._getF(self.rho_a2)
        self.Q_a3 = self._getQ(self.rho_a2)
        self.SG_a3 = self._getSG(self.rho_a2)
        self.SW_a2 = self._getSW(self.rho_a2)

    def update(self):
        """Update field vars with (lect 16 sl 12)."""
        self._setPredictedVars()
        self._setCorrectedVars()
        self.elapsed += self.dt
        self.nsteps += 1

    # -------------------- generate and save movie ----------------------
    def dump(self):
        """Dump info to lists"""
        # primitives
        self.rho_list.append(self.getRho())
        self.p_list.append(self.getP())
        self.vx_list.append(self.getVx())
        self.vz_list.append(self.getVz())

        # conservatives
        self.G_list.append(self.G_a3.copy())
        self.W_list.append(self.W_a2.copy())

        # debugging / sanity check
        self.omega_list.append(self.getOmega())  # vorticity
        self.mass_list.append(self.getMass())
        self.checkConservation()

        # dumps
        self.nextdump += self.tdump
        self.tdump_list.append(self.elapsed)
        self.ndumps += 1

    def generateFrames(self):
        """Evolve system, dump at every tdump, stop at ttot."""
        while self.elapsed <= self.ttot:
            if self.elapsed >= self.nextdump:
                print('time elapsed = {:.3g} s'.format(self.elapsed))
                self.dump()

            self.update()

            # numerical instability leads to p < 0, making dt imaginary
            try:
                self._setdt()
            except RuntimeWarning:
                print('imaginary time step')
                print('  last dt = {:.6g} s'.format(self.dt))

                rhomin = self.rho_a2.min()
                rhomax = self.rho_a2.max()
                rhomini_a1, rhominj_a1 = where(self.rho_a2 == rhomin)
                rhomaxi_a1, rhomaxj_a1 = where(self.rho_a2 == rhomax)
                rhominCoords_list = [(i, j) for i,j
                        in zip(rhomini_a1, rhominj_a1)]
                rhomaxCoords_list = [(i, j) for i,j
                        in zip(rhomaxi_a1, rhomaxj_a1)]

                print('  rhomin = {:.6g} kg/m^3\n    at {},'
                    '\n  rhomax = {:.6g} kg/m^2\n    at {}'.format(
                        rhomin, rhominCoords_list, rhomax, rhomaxCoords_list))

                pmin = self.p_a2.min()
                pmax = self.p_a2.max()
                pmini_a1, pminj_a1 = where(self.p_a2 == pmin)
                pmaxi_a1, pmaxj_a1 = where(self.p_a2 == pmax)
                pminCoords_list = [(i, j) for i,j
                        in zip(pmini_a1, pminj_a1)]
                pmaxCoords_list = [(i, j) for i,j
                        in zip(pmaxi_a1, pmaxj_a1)]

                print('  pmin = {:.6g} Pa\n    at {},'
                    '\n  pmax = {:.6g} Pa\n    at {}'.format(
                    pmin, pminCoords_list, pmax, pmaxCoords_list))

                print('  vxmin = {:.6g}, vxmax = {:.6g} m/s'.format(
                    self.v_a3[0].min(), self.v_a3[0].max()))
                print('  vzmin = {:.6g}, vzmax = {:.6g} m/s'.format(
                    self.v_a3[1].min(), self.v_a3[1].max()))
                print('  Gxmin = {:.6g}, Gxmax = {:.6g} kg m/s'.format(
                    self.G_a3[0].min(), self.G_a3[0].max()))
                print('  Gzmin = {:.6g}, Gzmax = {:.6g} kg m/s'.format(
                    self.G_a3[1].min(), self.G_a3[1].max()))
                print('  Wmin = {:.6g}, Wmax = {:.6g} J'.format(
                    self.W_a2.min(), self.W_a2.max()))
                break

            # dump at last frame
            if self.ttot < self.elapsed:
                print('Reached ttot successfully. Nice job!')
                break

        self.dump()
        print('time elapsed = {:.6g} s'.format(self.elapsed))
        print('nsteps = {}'.format(self.nsteps))
        print('ndumps = {}'.format(self.ndumps))

    def _generateAnimation(self):
        """Initialize AxesImage instance for field variable heatmap."""
        if self.var[0]=='r' or self.var[0]=='R':
            var_list = self.rho_list
            label = r"$\rho$ (kg/m$^3$)"
        elif self.var[0]=='p' or self.var[0]=='P':
            var_list = self.p_list
            label = "$p$ (Pa)"
        elif 'x' in self.var:
            var_list = self.vx_list
            label = "$v_x$ (m/s)"
        elif 'z' in self.var:
            var_list = self.vz_list
            label = "$v_z$ (m/s)"
        else:
            print('unrecognized variable')
            return

        # collect time-evolved field variables
        var_a3 = array(var_list)

        # imshow plots elements in reverse with first index on vertical axis
        self.var_a3 = var_a3.transpose((0, 2, 1))[:, ::-1, ::-1]

        # heatmap
        self.fig, self.ax = plt.subplots(figsize=(3.5, 8))
        self.im = self.ax.imshow(self.var_a3[0],
                extent=[-self.xmax, self.xmax, -self.d, self.d],
                cmap='plasma', interpolation='nearest')

        # colorbar
        cb = self.fig.colorbar(self.im)
        cb.set_label(label=label, fontsize=16)

        # text: frame number and mass
        self.title = self.ax.text(0, 1.01,
                'frame 0\nmass = {:.5g} kg'.format(self.mass_list[0]),
                ha='left', va='bottom', transform=self.ax.transAxes)
        self.ax.set_xlabel('x (m)', fontsize=16)
        self.ax.set_ylabel('z (m)', fontsize=16)

        plt.tight_layout()

    def _animate(self, n):
        """Prepare matplotlib image of current timestep.

        n: timestep index
        """
        var_a2 = self.var_a3[n]
        mass = self.mass_list[n]
        t = self.tdump_list[n] * 1000  # ms
        self.im.set_data(var_a2)
        self.im.set_clim(var_a2.min(), var_a2.max())
        self.title.set_text('t = {:>8.3f} ms\nmass = {:.5g} kg'.format(t, mass))

    def saveAnimation(self, var='rho', outfile='movie.mp4', interval=500):
        """Run simulation and save movie.

        var: variable to be plotted, e.g. p, rho, vz, or vx (str)
        outfile: name of movie file (str)
        interval: delay between frames in milliseconds (pos int)
        """
        self.var = var

        # add variable name to outfile
        name, ext = outfile.split('.')
        outfile = '{}_{}.{}'.format(name, var, ext)

        self._generateAnimation()
        ani = FuncAnimation(self.fig, self._animate, frames=self.ndumps,
                interval=interval)
        ani.save(outfile, 'ffmpeg')
        plt.show()
        
    def plot_conservation(self):
        """ Plots Mass, Energy, and Momentum conservation errors """
        
        plt.plot([100*(x - self.Mass_i)/self.Mass_i
            for x in self.mass_list])
        plt.title("Mass Error")
        plt.xlabel("Iteration")
        plt.ylabel("Percent Error")
        plt.show()
        
        plt.plot([100*(x - self.Energy_i)/self.Energy_i
            for x in self.energy_list])
        plt.title("Energy Error")
        plt.xlabel("Iteration")
        plt.ylabel("Percent Error")
        plt.show()


def runSimulation(Nt=1e4, Nx=200, Nz=200):
    """Returns field variables for all simulation time steps.

    Nt: number of time steps (pos int)
    """
    # initialize RT fluid instance
    f = Fluid()

    # holders for field variable data throughout simulation
    v_a4 = zeros((Nt, 2, Nx, Nz))
    rho_a3 = zeros((Nt, Nx, Nz))
    p_a3 = zeros((Nt, Nx, Nz))

    for n in range(Nt):
        v_a4[n] = f.getV()
        rho_a3[n] = f.getRho()
        p_a3[n] = f.getP()

        f.update()

    return v_a4, rho_a3, p_a3


# if __name__ == '__main__':
#     main()

#---------------------------------- SCRATCH:--------------------------------

def getLogistic(z, k=1, rhoh=1, rhol=0):
    """
    diffuse density profile
    """
    return (rhoh-rhol) / (1 + exp(-k*z)) + rhol

def getHeaviside(z, rhoh=1, rhol=0):
    """
    sharp density profile
    """
    return where(z>0, rhoh, rhol)

# predictor step over all grid points (or as many as possible)
#         self.rho_a2[:-1,:-1] = self.rho_a2[:-1,:-1] \
#             + self.dt * ( \
#                 - (self.G_a3[0,1:,:-1] - self.G_a3[0,:-1,:-1]) / self.dx \
#                 - (self.G_a3[1,:-1,1:] - self.G_a3[1,:-1,:-1]) / self.dz \
#             )
#     
#         self.G_a3[:,:-1,:-1] = self.G_a3[:,:-1,:-1] \
#             + self.dt * ( \
#                 - (self.F_a4[0,:,1:,:-1] - self.F_a4[0,:,:-1,:-1]) / self.dx \
#                 - (self.F_a4[1,:,:-1,1:] - self.F_a4[1,:,:-1,:-1]) / self.dz \
#                 + self.SG_a3[:,:-1,:-1]
#             )
# 
#         self.W_a2[:-1,:-1] = self.W_a2[:-1,:-1] \
#             + self.dt * ( \
#                 - (self.Q_a3[0,1:,:-1] - self.Q_a3[0,:-1,:-1]) / self.dx \
#                 - (self.Q_a3[1,:-1,1:] - self.Q_a3[1,:-1,:-1]) / self.dz \
#                 + self.SW_a2[:-1,:-1]
#             )

# predictor step restricted to "computational domain" (typed page 2)
#         self.rho_a2[1:-1,1:-1] = self.rho_a2[1:-1,1:-1] + self.dt * ( \
#             - (self.G_a3[0,2:,1:-1] - self.G_a3[0,1:-1,1:-1]) / self.dx \
#             - (self.G_a3[1,1:-1,2:] - self.G_a3[1,1:-1,1:-1]) / self.dz \
#             )
#     
#         self.G_a3[:,1:-1,1:-1] = self.G_a3[:,1:-1,1:-1] + self.dt * ( \
#             - (self.F_a4[0,:,2:,1:-1] - self.F_a4[0,:,1:-1,1:-1]) / self.dx \
#             - (self.F_a4[1,:,1:-1,2:] - self.F_a4[1,:,1:-1,1:-1]) / self.dz \
#             + self.SG_a3[:,1:-1,1:-1]
#             )
# 
#         self.W_a2[1:-1,1:-1] = self.W_a2[1:-1,1:-1] + self.dt * ( \
#             - (self.Q_a3[0,2:,1:-1] - self.Q_a3[0,1:-1,1:-1]) / self.dx \
#             - (self.Q_a3[1,1:-1,2:] - self.Q_a3[1,1:-1,1:-1]) / self.dz \
#             + self.SW_a2[1:-1,1:-1]
#             )

# p0(z) from analytical integration
#         self.p0_a1 = self.pd + self.g / 2 * ( \
#                 (self.rhoh-self.rhol) * self.L0 \
#                 * log((cosh(self.d/self.L0) / cosh(self.z_a1/self.L0))) \
#                 + (self.rhoh+self.rhol) * (self.d-self.z_a1)
#                 )

# p0(z) from forward difference
#         pdiff = (self.rhoh + self.rhol)*self.d*self.g
#         p0_list = [self.pd + pdiff]
#         for j, rho in enumerate(self.rho0_a1[:-1]):
#             p0j1 = p0_list[-1] - rho*self.g*self.dz
#             p0_list.append(p0j1)
#         self.p0_a1 = array(p0_list)

# p0(z) from central difference
#         pdiff = (self.rhoh + self.rhol)*self.d*self.g
#         p0_list = [self.pd + pdiff]
#         for j in range(self.Nz-1):
#             p0j1 = p0_list[-1] - 0.5*(self.rho0_a1[j+1]+self.rho0_a1[j]) \
#                     *self.g*self.dz
#         self.p0_a1 = array(p0_list)

