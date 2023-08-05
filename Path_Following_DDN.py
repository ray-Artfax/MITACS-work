from time import time
import casadi as ca
import numpy as np
from casadi import sin, cos, pi
import matplotlib.pyplot as plt
from simulation_codelol import simulate

Q_x = 1
Q_y = 5
Q_theta = 0.1
R1 = 0.5
R2 = 0.05


step_horizon = 0.5  # time between steps in seconds
N = 20              # number of look ahead steps
sim_time = 50     # simulation time

x_init = 0
y_init = 0
theta_init = 0

v_max = 1
v_min = -1

def shift_timestep(step_horizon, t0, state_init, u, f):
    f_value = f(state_init, u[:, 0])
    next_state = ca.DM.full(state_init + (step_horizon * f_value))

    t0 = t0 + step_horizon
    u0 = ca.horzcat(
        u[:, 1:],
        ca.reshape(u[:, -1], -1, 1)
    )

    return t0, next_state, u0


def DM2Arr(dm):
    return np.array(dm.full())


# state symbolic variables
x = ca.SX.sym('x')
y = ca.SX.sym('y')
theta = ca.SX.sym('theta')
states = ca.vertcat(
    x,
    y,
    theta
)
n_states = states.numel()

# control symbolic variables
V = ca.SX.sym('V')
omega= ca.SX.sym('omega')
controls = ca.vertcat(
    V,
    omega
)
n_controls = controls.numel()

# matrix containing all states over all time steps +1 (each column is a state vector)
X = ca.SX.sym('X', n_states, N + 1)

# matrix containing all control actions over all time steps (each column is an action vector)
U = ca.SX.sym('U', n_controls, N)

# coloumn vector for storing initial state and target state
P = ca.SX.sym('P', n_states + N * (n_states + n_controls)+1)

Q = ca.diagcat(Q_x, Q_y, Q_theta)

R = ca.diagcat(R1, R2 )

# discretization model (e.g. x2 = f(x1, v, t) = x1 + v * dt)

RHS = ca.vertcat(
    V*cos(theta),
    V*sin(theta),
    omega
)
g=[]
f = ca.Function('f', [states, controls], [RHS])
st = X[:, 0]
g = ca.vertcat(g, st - P[0:3])
obj=0

for k in range(1,N+1):
    st = X[:, k-1]
    con = U[:, k-1]
    obj= obj \
        + (st - P[5 * k - 1:5 * k + 2]).T @ Q @ (st - P[5 * k - 1:5 * k + 2]) \
        + (con - P[5 * k + 2:5 * k + 4]).T @ R @ (con - P[5 * k + 2:5 * k + 4])
    '''obj = obj + ca.mtimes((st - P[5 * k - 1:5 * k + 2]).T, ca.mtimes(Q, (st - P[5 * k - 1:5 * k + 2]))) + ca.mtimes(
        (con - P[5 * k + 2:5 * k + 4]).T, ca.mtimes(R, (con - P[5 * k + 2:5 * k + 4])))'''
    
    st_next = X[:, k ]
    k1 = f(st, con)
    k2 = f(st + step_horizon/2*k1, con)
    k3 = f(st + step_horizon/2*k2, con)
    k4 = f(st + step_horizon * k3, con)
    st_next_RK4 = st + (step_horizon / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    g = ca.vertcat(g, st_next - st_next_RK4)



OPT_variables = ca.vertcat(
    X.reshape((-1, 1)),   # Example: 3x11 ---> 33x1 where 3=states, 11=N+1
    U.reshape((-1, 1))
)
nlp_prob = {
    'f': obj,
    'x': OPT_variables,
    'g': g,
    'p': P
}

opts = {
    'ipopt': {
        'max_iter': 2000,
        'print_level': 0,
        'acceptable_tol': 1e-8,
        'acceptable_obj_change_tol': 1e-6
    },
    'print_time': 0
}

solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

lbx = ca.DM.zeros((n_states*(N+1) + n_controls*N, 1))
ubx = ca.DM.zeros((n_states*(N+1) + n_controls*N, 1))

lbx[0: n_states*(N+1): n_states] = -20  # X lower bound
lbx[1: n_states*(N+1): n_states] = -20     # Y lower bound
lbx[2: n_states*(N+1): n_states] = -ca.inf     # theta lower bound

ubx[0: n_states*(N+1): n_states] = 20     # X upper bound
ubx[1: n_states*(N+1): n_states] = 20     # Y upper bound
ubx[2: n_states*(N+1): n_states] = ca.inf      # theta upper bound

lbx[n_states*(N+1):] = v_min                  # v lower bound for all V
ubx[n_states*(N+1):] = v_max                  # v upper bound for all V


args = {
    'lbg': ca.DM.zeros((n_states*(N+1), 1)),  # constraints lower bound
    'ubg': ca.DM.zeros((n_states*(N+1), 1)),  # constraints upper bound
    'lbx': lbx,
    'ubx': ubx
}

t0 = 0
state_init = ca.DM([x_init, y_init, theta_init])        # initial state
#state_target = ca.DM([x_target, y_target, theta_target])  # target state

# xx = DM(state_init)
t = ca.DM(t0)

u0 = ca.DM.zeros((n_controls, N))  # initial control
X0 = ca.repmat(state_init, 1, N+1)         # initial state full

mpc_iter = 0
cat_states = DM2Arr(X0)
cat_controls = DM2Arr(u0[:, 0])
times = np.array([[0]])


###############################################################################

if __name__ == '__main__':
    main_loop = time()  # return time in sec
    while (mpc_iter * step_horizon < sim_time):
        current_time = mpc_iter * step_horizon
        t1 = time()
        args['p'] = np.concatenate((state_init , np.zeros((N * (n_states + n_controls)+1, 1))))
        # optimization variable current state
        args['x0'] = ca.vertcat(
            ca.reshape(X0, n_states*(N+1), 1),
            ca.reshape(u0, n_controls*N, 1)
        )
        for k in range(1,N+1):
            t_predict = current_time + (k - 1) * step_horizon
            x_ref = 0.1 * pi * t_predict
            y_ref = sin(0.1 * pi * t_predict)
            theta_ref = 0
            u_ref = 1
            omega_ref = 0
            if x_ref >= 4 * pi:
                x_ref = 4 * pi
                y_ref = 0
                theta_ref = 0
                u_ref = 0
                omega_ref = 0
            args['p'][5 * k - 1:5 * k + 2] = np.array([x_ref, y_ref, theta_ref]).reshape(3,1)
           
            args['p'][5 * k + 2:5 * k + 4] = np.array([u_ref, omega_ref]).reshape(2,1)
        sol = solver(
            x0=args['x0'],
            lbx=args['lbx'],
            ubx=args['ubx'],
            lbg=args['lbg'],
            ubg=args['ubg'],
            p=args['p']
        )

        u = ca.reshape(sol['x'][n_states * (N + 1):], n_controls, N)
        X0 = ca.reshape(sol['x'][: n_states * (N+1)], n_states, N+1)
        #print(X0)

        cat_states = np.dstack((
            cat_states,
            DM2Arr(X0)
        ))

        cat_controls = np.vstack((
            cat_controls,
            DM2Arr(u[:, 0])
        ))
        t = np.vstack((
            t,
            t0
        ))

        t0, state_init, u0 = shift_timestep(step_horizon, t0, state_init, u, f)

        # print(X0)
        X0 = ca.horzcat(
            X0[:, 1:],
            ca.reshape(X0[:, -1], -1, 1)
        )

        # xx ...
        t2 = time()
        #print(mpc_iter)
        #print(t2-t1)
        times = np.vstack((
            times,
            t2-t1
        ))

        mpc_iter = mpc_iter + 1

    main_loop_time = time()
    #ss_error = ca.norm_2(state_init - state_target)

    print('\n\n')
    print('Total time: ', main_loop_time - main_loop)
    print('avg iteration time: ', np.array(times).mean() * 1000, 'ms')
    #print('final error: ', ss_error)
    #print(cat_states)
    #print(cat_controls)
    # simulate
    simulate(cat_states, cat_controls, times, step_horizon, N,
             np.array([x_init, y_init, theta_init]), save=False)
