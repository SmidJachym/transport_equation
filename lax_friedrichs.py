import numpy as np
import time
import sys
import os

""" 
TODO:
    čas animace (oprava)
    upravit write data - one file: jeden řádek = jeden časový krok
    write bigger chunks
"""

# --------------------------------- CALCULATION -------------------------------------

def rect_grid(x_l, y_l, dx, dy):     # generate a rectangular grid
    num_x = int((x_l[1] - x_l[0])/dx)
    num_y = int((y_l[1] - y_l[0])/dy)
    x = np.linspace(x_l[0], x_l[1], num_x)
    y = np.linspace(y_l[0], y_l[1], num_y)
    X, Y = np.meshgrid(x, y, indexing='ij')
    return [X,Y]

def initial_conditions(grid):    # calculate and impose initial conditions
    X = grid[0]
    Y = grid[1]
    
    U = 100*np.cos(((X+1)**2+Y**2)/0.1)
    return U

def boundary_conditions(grid, U):    # calculate and the boundary conditions
    X = grid[0]
    Y = grid[1]

    U[:,0] = 0.0
    U[:,-1] = 0.0
    U[0,:] = 0.0
    U[-1,:] = 0.0


def my_LF(grid, U_p, p, dx, dy, dt):    # my implementation of Lax-Friedrichs scheme
    X = grid[0]
    Y = grid[1]
    U_n = U_p.copy()
    for i in range(1,len(X)-1):
        for j in range(1,len(Y)-1):
            U_n[i, j] = 1/4 * (U_p[i+1, j] + U_p[i-1, j] + U_p[i, j+1] + U_p[i, j-1]) - \
                dt * ((U_p[i+1, j] - U_p[i-1, j]) / (2 * dx) - 2 * p * X[i,j] * (U_p[i, j+1] - \
                U_p[i, j-1]) / (2 * dy))
    return U_n

# ------------------------------ POSTPROCESSING -------------------------------------

def write_data(memmap_object,start, end, data):
    if len(data.shape) < 3:
        memmap_object[:,:,0] = data[:,:]
        return

    memmap_object[:,:,start:end] = data[:,:,:]
    memmap_object.flush()

# ------------------------------------------ MAIN --------------------------------------------

def main():

    # define domain
    x_interval = [-1.5, 1.5]
    y_interval = [-1.5, 1.5]
    x_int_len = x_interval[1]-x_interval[0]
    y_int_len = y_interval[1]-y_interval[0]

    # Define parametres
    p = 1.0
    step_x = 0.03
    step_y = step_x
    step_t = min(step_x/2, step_y/(4*abs(p)*max(*np.abs(x_interval))))
    T = 2.0

    if int(np.ceil(T/step_t)) < 300: # necessary for animation
        num_t_steps = 300
    else:
        num_t_steps = int(np.ceil(T/step_t))
    
    # Create grid
    my_grid = rect_grid(
        x_l=x_interval,
        y_l=y_interval,
        dx= step_x,
        dy= step_y)

    # memmap object for data storage
    data_memmap = np.memmap(rf"data/data{int(x_int_len/step_x)}x{int(y_int_len/step_y)}.dat", dtype='float32', mode='w+', shape=(int(x_int_len/step_x), int(y_int_len/step_y), num_t_steps+1))

    # Impose initial values and store them
    U_prev = initial_conditions(my_grid)
    write_data(data_memmap, 0, 0, U_prev)

    # start timer
    start_time = time.time()

    # report
    print(f"\nCreating grid:\n\t- x steps: {x_int_len/step_x}\n\t- y steps: {y_int_len/step_y}")
    print(f"Number of grid nodes: {x_int_len/step_x*y_int_len/step_y}")
    print(f"\rCalculating {num_t_steps} steps")
    print(f"\t- with time step dt = {step_t}")
    print(f"Total number of calculations: {x_int_len/step_x*y_int_len/step_y*num_t_steps}\n")

    # main loop
    k = 0
    n = 1
    U_store = np.zeros((int(x_int_len/step_x), int(y_int_len/step_y), 100))
    for i in range(num_t_steps+1):
        if i == 100*n:
            write_data(data_memmap, i-99, i+1, U_store)
            k = 0
            n += 1
        if i == num_t_steps:
            break

        # Calcualte the next step
        U_next = my_LF(
            grid=my_grid,
            U_p=U_prev,
            p=p,
            dx=step_x,
            dy=step_y,
            dt=step_t
            )

        # Impose boundary conditions
        boundary_conditions(
            grid=my_grid,
            U=U_next
            )

        # Copy the "next" array into "previous"
        U_prev = U_next.copy()
        U_store[:,:,k] = U_prev[:,:]

        U_store[:,:,k] = U_prev[:,:]
        k += 1
        
        # track progress - report
        sys.stdout.write(f"\rProgress: {i+1}/{num_t_steps}    :    {int(((i+1)*100/num_t_steps)*100)/100}%     ")
        sys.stdout.flush()

    # end timer
    end_time = time.time()
    elapsed_time = end_time - start_time

    # report
    print(f"Elapsed time: {int(elapsed_time/3600)}:{int((elapsed_time-int(elapsed_time/3600)*3600)/60)}:{elapsed_time - int(elapsed_time/3600)*3600 - int(elapsed_time/60)*60}")
    print(f"Time per calculation: {elapsed_time/num_t_steps*1000} ms\n")

if __name__ == "__main__":
    main()
