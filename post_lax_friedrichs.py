import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["animation.ffmpeg_path"] = "usr/bin/ffmpeg"
from matplotlib import cm
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation 


x_interval = [-1.5, 1.5]
y_interval = [-1.5, 1.5]
x_int_len = x_interval[1] - x_interval[0]
y_int_len = y_interval[1] - y_interval[0]

p = 1.0
step_x = 0.03
step_y = step_x
step_t = min(step_x / 2, step_y / (4 * abs(p) * max(np.abs(x_interval))))
T = 2.0

if int(np.ceil(T / step_t)) < 300:
    num_t_steps = 300
else:
    num_t_steps = int(np.ceil(T / step_t))


def rect_grid(x_l, y_l, dx, dy):
    num_x = int((x_l[1] - x_l[0]) / dx)
    num_y = int((y_l[1] - y_l[0]) / dy)
    x = np.linspace(x_l[0], x_l[1], num_x)
    y = np.linspace(y_l[0], y_l[1], num_y)
    X, Y = np.meshgrid(x, y, indexing="ij")
    return [X, Y]


def read_data(memmap_object, time_step):
    return memmap_object[:, :, time_step]


def value_loss(memmap_object, num_t_steps):
    x = np.linspace(0, T, num_t_steps)
    y = np.zeros(len(x))
    for i in range(num_t_steps):
        y[i] = np.sum(read_data(memmap_object, i))

    plt.plot(x, y, color="crimson", lw=2.25)
    plt.title("Loss of U over time")
    plt.xlabel("t")
    plt.ylabel(r"$\sum U$")
    plt.grid()
    plt.savefig(rf"graphs/u_loss_{step_x}x{step_y}")


def visualize_2d(grid, memmap_object, step, name):
    X = grid[0]
    Y = grid[1]
    Z = read_data(memmap_object, step)

    x_vector = np.linspace(-1.5, 1.5, 15)
    x, y = np.meshgrid(x_vector, x_vector)

    u = 1.0
    v = -2.0 * x
    
    plt.figure()
    plt.contourf(X, Y, Z, cmap="Reds")
    plt.colorbar()
    plt.quiver(x, y, u, v, color="salmon")
    plt.title(name)
    plt.grid()
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.savefig(rf"graphs/{name}.pdf")


def visualize_3d(grid, step, name, memmap_object):
    X = grid[0]
    Y = grid[1]
    Z = read_data(memmap_object, step)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(X, Y, Z, color="gray")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(name)

    plt.savefig(rf"graphs/{name}.pdf")


def animate_3d(grid, num_t, dt, memmap_object):
    x = grid[0]
    y = grid[1]
    z = read_data(memmap_object, 0)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    n = int(np.round(num_t / 300, 0))
    ax.plot_surface(x, y, z, cmap="Reds")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("U")
    ax.set_title("t = 0")
    ax.set_zlim([0, 130])

    def update(frame):
        ax.cla()  # clear all
        ax.set_title(f"t = {np.ceil(n*(frame+1)*dt*100)/100}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("U")
        ax.set_zlim([0, 130])

        if n * (frame + 1) > num_t:
            return
        z = read_data(memmap_object, n * (frame + 1))
        ax.plot_surface(x, y, z, cmap="Reds")

        if np.ceil(n * (frame + 1) * dt * 100) / 100 > 2.0:
            return

    ani = FuncAnimation(fig, update, frames=np.arange(0, 300), interval=33)
    FFwriter = animation.FFMpegWriter()
    ani.save(r"graphs/animation.gif", dpi=300, writer=FFwriter)


def visualize_section(grid, section, memmap_object, t, dx, dy, dt, axis):

    Y = read_data(memmap_object, t)
    plt.figure()

    if axis == "x":
        X = grid[0]
        X = X[:, 0]
        plt.plot(X, Y[:, section], color="crimson", lw=2.25)
        plt.xlabel("x")
        plt.ylabel(f"u(x,y = {np.round(section*dy + grid[1][0,0],1)}, t = {t*dt})")
        plt.grid()
        plt.savefig(
            rf"graphs/u(x,y={np.round(section*dy + grid[1][0,0],1)},t={t*dt}).pdf"
        )

    elif axis == "y":
        X = grid[1]
        X = X[0, :]
        plt.plot(X, Y[section, :], color="crimson", lw=2.25)
        plt.xlabel("y")
        plt.ylabel(f"u(x = {np.round(section*dx + grid[0][0,0],1)}, y, t = {t*dt})")
        plt.grid()
        plt.savefig(
            rf"graphs/u(x,y={np.round(section*dy + grid[0][0,0],1)},t={t*dt}).pdf"
        )


def main():

    data_memmap = np.memmap(
        rf"data/data{int(x_int_len/step_x)}x{int(y_int_len/step_y)}.dat",
        dtype="float32",
        mode="r",
        shape=(int(x_int_len / step_x), int(y_int_len / step_y), num_t_steps + 1),
    )

    my_grid = rect_grid(x_l=x_interval, y_l=y_interval, dx=step_x, dy=step_y)

    animate_3d(grid=my_grid, num_t=num_t_steps, dt=step_t, memmap_object=data_memmap)

    for i in range(5):
        visualize_2d(
            grid=my_grid,
            step=int(i / (2 * step_t)),
            name=f"u(x,y, t = {i/2})",
            memmap_object=data_memmap,
        )

    visualize_section(
        grid=my_grid,
        section=int(np.round((0 - x_interval[0]) / step_x, 0)),
        t=0,
        dx=step_x,
        dy=step_y,
        dt=step_t,
        axis="x",
        memmap_object=data_memmap,
    )
    visualize_section(
        grid=my_grid,
        section=int(np.round((-1 - y_interval[0]) / step_y, 0)),
        t=0,
        dx=step_x,
        dy=step_y,
        dt=step_t,
        axis="y",
        memmap_object=data_memmap,
    )
    visualize_section(
        grid=my_grid,
        section=int(np.round((0 - x_interval[0]) / step_x, 0)),
        t=num_t_steps,
        dx=step_x,
        dy=step_y,
        dt=step_t,
        axis="x",
        memmap_object=data_memmap,
    )
    visualize_section(
        grid=my_grid,
        section=int(np.round((1 - y_interval[0]) / step_y, 0)),
        t=num_t_steps,
        dx=step_x,
        dy=step_y,
        dt=step_t,
        axis="y",
        memmap_object=data_memmap,
    )

    value_loss(memmap_object = data_memmap, num_t_steps = num_t_steps)


if __name__ == "__main__":
    main()
