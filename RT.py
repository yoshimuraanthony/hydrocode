from numpy import array, arange, zeros, ones, linspace, newaxis
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
    """Discritized classical 2D RT fluid with periodic boundary conditions

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
            tdump=.001, ttot=.1, tscale = 1,  # s
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

        # initial primitive field variables
        self.rho_a2 = self.rho0_a1 + zeros((Nx, Nz))
        self.p_a2 = self.p0_a1 + zeros((Nx, Nz))
        self.v_a3 = zeros((2, Nx, Nz))  # 2D vector field
#         self.perturb()
        
        # initial conservative field variables
        self.pi_a2 = zeros((Nx, Nz)) # z-derivative of pressure
        self._setG()
        self._setF()
        self._setW()
        self._setQ()
        self._setSG()
        self._setSW()
#         self._enforceBCs()

        # time step and time elapsed
        self.tscale = tscale
        self._setdt()
        self.elapsed = 0.
        self.tdump = tdump
        self.ndumps = 0
        self.nextdump = 0.
        self.ttot = ttot
        self.nsteps = 0

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
        Mass_i = self.getMass()
        Energy_i = self.getEnergy()
#         Momentum_i = self.getMomentum()  # component
        # corrected conservatives and predicted primitives should be set in
        #    _setPredictedVars
        # predicted conservatives and corrected primitives should be set in
        #    _setCorrectedVars
#        # predicted variables
#         self.pv_a3 = self.v_a3.copy()
#         self.prho_a2 = self.rho_a2.copy() 
#         self.pp_a2 = self.p_a2.copy()
#         self.pG_a3 = self.G_a3.copy()
#         self.pF_a4 = self.F_a4.copy()
#         self.pW_a2 = self.W_a2.copy()
#         self.pQ_a3 = self.Q_a3.copy()
#         self.psG_a3 = self.sG_a3.copy()
#         self.psW_a2 = self.sW_a2.copy()

    # ---------------------- initialize primitvies -----------------------
    def setRho0(self):
        """Set equilibrium rho0(z) (lec 16 sl 4)."""
        self.rho0_a1 = (self.rhoh - self.rhol) \
                * (tanh(self.z_a1/self.L0) + 1) \
                / 2 + self.rhol

    def setP0(self):
        """Set equilibrium p0(z) = p0(d) + Int(rho0, z) (lec 16 sl 4)."""
        pdiff = (self.rhoh + self.rhol)*self.d*self.g
        p0_list = [self.pd + pdiff]
        for j, rho in enumerate(self.rho0_a1[:-1]):
            p0j1 = p0_list[-1] - rho*self.g*self.dz
            p0_list.append(p0j1)
        self.p0_a1 = array(p0_list)

    def perturb(self):
        """Perturb the field variables from equilibrium."""
        k = pi/self.xmax
        x_a2 = self.x_a1[:, newaxis]
        z_a2 = self.z_a1[newaxis, :]
        self.v_a3[0] = self.vpert*sin(k*x_a2) \
                * exp(-k*abs(tanh(z_a2/self.L0)*z_a2)) \
                * tanh(z_a2)
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
            var_a2 = self.v_a3[0]
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

    # ----------------------- set field variblse -------------------------
    def _setv(self):
        """Set velocity from conservatives (typed eq 11)."""
        self.v_a3 = self.G_a3 / self.rho_a2

    def _setp(self):
        """Set pressure from conservatives (typed eq 11)."""
        self.p_a2 = (self.gamma - 1) \
                * (self.W_a2 - (self.G_a3**2).sum(0)/self.rho_a2/2)

    def _setG(self):
        """Set momentum from primitives (lec 16 sl 14)."""
        self.G_a3 = self.rho_a2 * self.v_a3

    def _setF(self):
        """Set momentum flux from primitives (lec 16 sl 14)."""
        self.F_a4 =  self.rho_a2[newaxis, newaxis, :, :] \
                * self.v_a3[newaxis, :, :, :] \
                * self.v_a3[:, newaxis, :, :]
        self.F_a4[0, 0] += self.p_a2
        self.F_a4[1, 1] += self.p_a2

    def _setW(self):
        """Set energy from primitives (lec 16 sl 14)."""
        self.W_a2 = self.p_a2 / (self.gamma-1) \
                + self.rho_a2 * (self.v_a3**2).sum(0) / 2

    def _setQ(self):
        """Set energy flux from primitives (lec 16 sl 14)."""
        self.Q_a3 = self.v_a3 \
                * (self.gamma*self.p_a2 / (self.gamma-1) \
                + self.rho_a2 * (self.v_a3**2).sum(0) / 2)

    def _setSG(self):
        """Set momentum source from primitives (lec 16 sl 17)."""
        self.SG_a3 = zeros((2, self.Nx, self.Nz))
        self.SG_a3[1] = -self.g * self.rho_a2  # g is nonzero only in z-direction

    def _setSW(self):
        """Set energy source from primitives (lec 16 sl 17)."""
        self.SW_a2 = -self.g * self.rho_a2 * self.v_a3[1]

    def _setPi(self):
        """Return derivative of pressure in z"""
        Gz_a2 = self.v_a3[1,:,:]*self.rho_a2
        self.pi_a2 = self.p_a2 + Gz_a2**2/self.rho_a2

    def _setdt(self):
        """Set time step in seconds."""
        cs_a2 = sqrt(self.gamma*self.p_a2/self.rho_a2)
        vmag_a2 = sqrt((self.v_a3**2).sum(0))
        self.dt = self.dmin / (cs_a2 + vmag_a2).max() / 2 * self.tscale

    # ------------------ enforce boundary conditions ---------------------
    def _enforceBCs(self):
        """Enforce boundary conditions. Reflective Boundary Conditions"""
        # Define some commonly used variables
        Nx = self.Nx 
        Nz = self.Nz
        g  = self.g
        dz = self.dz
        gamma = self.gamma

        # Save initial mass, energy, and momentum
        Mass_i = self.getMass()
        Energy_i = self.getEnergy()
        Momentum_i = self.G_a3[0].sum() # apply for x-direction
        # variables altered by end of enforceBCs:

        # (1) self.v_a3, (2) self_W_a2, (3) self.rho_a2, and (4) self.pi_a2
        # Apply Reflective Boundary Conditions in the x-direction
        # i = 0 BCs
        self.G_a3[0,0,:] = -self.G_a3[0,1,:]
        self.G_a3[1,0,:] = self.G_a3[1,1,:]
        self.W_a2[0,:] = self.W_a2[1,:]
        self.rho_a2[0,:] = self.rho_a2[1,:]

        # i = Nx+1 BCs
        self.G_a3[0,-1,:] = -self.G_a3[0,-2,:]
        self.G_a3[1,-1,:] = self.G_a3[1,-2,:]
        self.W_a2[-1,:] = self.W_a2[-2,:]
        self.rho_a2[-1,:] = self.rho_a2[-2,:]

        # Apply Rigid Wall Boundary Conditions in the z-direction
        # j = 0 BCs
        self.G_a3[0,:,0] = -self.G_a3[0,:,1]
        self.G_a3[1,:,0] = self.G_a3[1,:,1]
        self.rho_a2[:,0] = self.rho_a2[:,1]
        self.pi_a2[:,0] = self.pi_a2[:,1] + self.rho_a2[:,1]*g*dz

        # w(i,0)  =  ... ###
        Gx2 = self.G_a3[0,:,0]**2
        Gz2 = self.G_a3[1,:,0]**2
        term1 = self.pi_a2[:,1] + self.rho_a2[:,1]*g*dz \
                - Gz2/self.rho_a2[:,0]
        term2 = Gx2 + Gz2 
        self.W_a2[:,0] = term1/(gamma - 1.) + term2/2./self.rho_a2[:,0]

        # j = Nz+1 BCs
        self.G_a3[0,:,-1] = -self.G_a3[0,:,-2]
        self.G_a3[1,:,-1] = self.G_a3[1,:,-2]
        self.rho_a2[:,-1] = self.rho_a2[:,-2]
        self.pi_a2[:,-1] = self.pi_a2[:,-2] + self.rho_a2[:,-2]*g*dz

        # w(i,-1)  =  ... ###
        Gx2 = self.G_a3[0,:,-1]**2
        Gz2 = self.G_a3[1,:,-1]**2
        term1 = self.pi_a2[:,-2] + self.rho_a2[:,-2]*g*dz \
                - Gz2/self.rho_a2[:,-1]
        term2 = Gx2 + Gz2 
        self.W_a2[:,-1] = term1/(gamma - 1.) + term2/2./self.rho_a2[:,-1]
            
    # ----------------------- check conservation -------------------------
    def getMass(self):
        """Return total mass in kg."""
        return self.rho_a2.sum()

    def getEnergy(self):
        """Return total energy in J."""
        return 1.

    def getOmega(self):
        """Return vorticity magnitude"""
        dvzdx_a2 = (self.v_a3[1,1:,:-1] - self.v_a3[1,:-1,:-1]) / self.dx
        dvxdz_a2 = (self.v_a3[0,:-1,1:] - self.v_a3[0,:-1,:-1]) / self.dz
        return abs(dvzdx_a2 - dvxdz_a2).sum()

    def checkConservation():
        """Check for Conservation of mass, energy, and momentum."""
        Mass_f = self.getMass()
        Energy_f = self.getEnergy()
        Momentum_f = self.G_a3[0].sum()

        if abs(Mass_f - Mass_i) < 1e-8:
            print('Total Mass is Conserved')
        else:
            print('Total Mass not Conserved by ', Mass_f - Mass_i)
        
        if abs(Energy_f - Energy_i) < 1e-8:
            print('Total Energy is Conserved')
        else:
            print('Total Energy not Conserved by ', Energy_f - Energy_i)
        
        if abs(Momentum_f - Momentum_i) < 1e-8:
            print('Total Momentum is Conserved')
        else:
           print('Total Momentum not Conserved by ', Momentum_f - Momentum_i)

    # -------------------- advance field varialbes -----------------------
    def _setPredictedVars(self):
        """Set "predictor" field variables.

        Update using information about momentum at time t (lect 16 sl 14).
        """
        # ACD: rewrite this with the self.pVars -- no return 
        # ACD: also corrector step for primitives 
        # ACD: make sure _set functions store COPIES of variables

        # advance field variables with forward differences
        self.rho_a2[:-1,:-1] = self.rho_a2[:-1,:-1] \
            + self.dt * ( \
                - (self.G_a3[0,1:,:-1] - self.G_a3[0,:-1,:-1]) / self.dx \
                - (self.G_a3[1,:-1,1:] - self.G_a3[1,:-1,:-1]) / self.dz \
            )
    
        self.G_a3[:,:-1,:-1] = self.G_a3[:,:-1,:-1] \
            + self.dt * ( \
                - (self.F_a4[0,:,1:,:-1] - self.F_a4[0,:,:-1,:-1]) / self.dx \
                - (self.F_a4[1,:,:-1,1:] - self.F_a4[1,:,:-1,:-1]) / self.dz \
                + self.SG_a3[:,:-1,:-1]
            )

        self.W_a2[:-1,:-1] = self.W_a2[:-1,:-1] \
            + self.dt * ( \
                - (self.Q_a3[0,1:,:-1] - self.Q_a3[0,:-1,:-1]) / self.dx \
                - (self.Q_a3[1,:-1,1:] - self.Q_a3[1,:-1,:-1]) / self.dz \
                + self.SW_a2[:-1,:-1]
            )
#         self._enforceBCs()

        # advance remaining primitives
        self._setv()
        self._setp()
#         self._enforceBCs()

        # advance remaining conservatives
        self._setF()
        self._setQ()
        self._setSG()
        self._setSW()
#         self._enforceBCs()

    def _setCorrectedVars(self):
        """Set "corrected" variables.

        Update field vars with avg time derivatives (lect 16 sl 15).
        """
        # ACD: move this to a predictor function
        # information about the spatial grid and current time step
        Nx = self.Nx 
        Nz = self.Nz 
        dt = self._setdt()

        # current field variables
        rho = self.getRho()     
        G = self._setG()        
        W = self._setW()        
        F = self._setF()
        Q = self._setQ() 

        # predicted field variables for the current time step 
        predict_rho, predict_G, predict_W = self._setPVars
        
        # ACD WIP
        rho[1:,1:] = 0.5 * ( (predict_rho[1:,1:] + rho[1:,1:]) - (dt/self.dx) * (predict_G[0,0:Nx-1,0:Nz-1] - predict_G[0,0:Nx-1,0:Nz-1]) )
        pass

    def update(self):
        """Update field vars with (lect 16 sl 12)."""
        # store animation frames
        if self.elapsed >= self.nextdump:

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

            # dumps
            self.nextdump += self.tdump
            self.tdump_list.append(self.elapsed)
            self.ndumps += 1

        self._setPredictedVars()
        self.elapsed += self.dt
        self.nsteps += 1

        # numerical instability leads to p < 0, making dt imaginary
        try:
            self._setdt()
        except RuntimeWarning:
            print('negative pressure')
            print('elapsed = {}'.format(self.elapsed))
            print('  nsteps = {}'.format(self.nsteps))
            print('  ndumps = {}'.format(self.ndumps))
            print('  dt = {}'.format(self.dt))
            print('  rhomax = {}, rhomin={}'.format(self.rho_a2.max(),
                self.rho_a2.min()))
            print('  pmax = {}, pmin={}'.format(self.p_a2.max(),
                self.p_a2.min()))
            print('  vxmax = {}, vxmin={}'.format(self.v_a3[0].max(),
                self.v_a3[0].min()))
            print('  vzmax = {}, vzmin={}'.format(self.v_a3[1].max(),
                self.v_a3[1].min()))
            print('  Gxmax = {}, Gxmin={}'.format(self.G_a3[0].max(),
                self.G_a3[0].min()))
            print('  Gzmax = {}, Gzmin={}'.format(self.G_a3[1].max(),
                self.G_a3[1].min()))
            print('  Wmax = {}, Wmin={}'.format(self.W_a2.max(),
                self.W_a2.min()))

    # -------------------- generate and save movie ----------------------
    def generateFrames(self):
        """Store primitives at every tdump."""
        while self.elapsed < self.ttot:
            self.update()

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

# p0(z) from central difference
#         for j in range(self.Nz-1):
#             p0j1 = p0_list[-1] - 0.5*(self.rho0_a1[j+1]+self.rho0_a1[j]) \
#                     *self.g*self.dz
