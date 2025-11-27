# jax imports
import jax
import jax.numpy as np
from jax import lax, vmap
from jax.tree_util import Partial

# Plotting imports
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.express.colors import qualitative

# from plotly_gif import GIF, capture
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Other imports
import numpy as onp
import sys

sys.path.append(".")
from sim_utils import calc_rdf

# jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_debug_nans", False)


def animate_trajectory(
    traj,
    es,
    d_fn,
    box_size,
    traj_species=None,
    nbrs_g=None,
    Ts=None,
    annotation_data=None,
):
    if callable(es):
        energies = vmap(es)(traj)
    else:
        energies = es

    xs = traj[:, :, 0]
    ys = traj[:, :, 1]
    zs = traj[:, :, 2] if traj.shape[2] > 2 else None

    print("Calculating G(r)")
    r_grs, gs = lax.map(
        lambda x: calc_rdf(
            d_fn, box_size, x[0], nbrs_g, species=x[1], r_max=10.0, n_bins=250, dr=0.05
        ),
        (traj, traj_species),
    )
    gs.block_until_ready()
    print(gs.shape)
    # gs = gs[:, 0, 0, :]
    print("G(r) done!")

    atom_plot_spec = {"type": "scatter3d"} if zs is not None else {}

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[atom_plot_spec, {}], [{"colspan": 2}, None]],
        vertical_spacing=0.1,
        horizontal_spacing=0.1,
    )

    def scatter_frame(i, xs=None, ys=None, zs=None, species=None, **kwargs):
        if zs is not None:
            return go.Scatter3d(
                x=xs[i],
                y=ys[i],
                z=zs[i],
                mode="markers",
                opacity=0.8,
                marker=dict(
                    size=5,
                    color=species[i],
                    # line=dict(width=1, color="DarkSlateGrey"),
                    # line=dict(width=1, color=species[i]),
                ),
                name="Atoms",
            )
        else:
            return go.Scatter(
                x=xs[i],
                y=ys[i],
                mode="markers",
                marker=dict(
                    size=5,
                    color=species[i],
                    # line=dict(width=1, color="DarkSlateGrey"),
                    # line=dict(width=1, color=species[i]),
                ),
                name="Atoms",
            )

    # Trajectory points frame [0]
    fig.add_trace(
        scatter_frame(0, xs=xs, ys=ys, zs=zs, species=traj_species),
        row=1,
        col=1,
    )
    # fig.add_trace(
    #     go.Scatter(
    #         x=xs[0],
    #         y=ys[0],
    #         mode="markers",
    #         marker=dict(
    #             size=5, color=traj_species[0], line=dict(width=2, color=traj_species[0])
    #         ),
    #         name="Atoms",
    #     ),
    #     row=1,
    #     col=1,
    # )

    # Energy background line
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(energies)),
            y=energies,
            line=dict(color="royalblue", width=1),
            name="Trajectory energy",
            customdata=Ts,
            hovertemplate="Frame: %{x:.3f}<br>Energy: %{y:.3f}<br>kT: %{customdata:.3f}",
        ),
        row=1,
        col=2,
    )

    # Energy point frame [0]
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[energies[0]],
            marker=dict(size=12, line=dict(width=2, color="DarkSlateGrey")),
            name="Energy of current frame",
        ),
        row=1,
        col=2,
    )

    # rdf partials frame [0]
    for s1 in range(gs.shape[1]):
        for s2 in range(gs.shape[1]):
            fig.add_trace(
                go.Scatter(
                    x=r_grs[0],
                    y=gs[0, s1, s2],
                    line=dict(
                        color=qualitative.Plotly[s1 * gs.shape[1] + s2],
                        width=4,
                    ),
                    name=f"[{s1}, {s2}] partial rdf",
                ),
                row=2,
                col=1,
            )
    # # rdf [0, 1] partial frame [0]
    # fig.add_trace(
    #     go.Scatter(
    #         x=r_grs[0],
    #         y=gs[0, 0, 1],
    #         line=dict(color="rgba(255, 102, 204, 1.0)", width=4),
    #         name="[0, 1] partial rdf of atoms in current frame",
    #     ),
    #     row=2,
    #     col=1,
    # )

    # define frames
    frames = [
        go.Frame(
            data=[
                scatter_frame(k, xs=xs, ys=ys, zs=zs, species=traj_species),
                go.Scatter(visible=True),
                go.Scatter(x=[k], y=[energies[k]]),
            ]
            + [
                go.Scatter(x=r_grs[k], y=gs[k, s1, s2])
                for s1 in range(gs.shape[1])
                for s2 in range(gs.shape[2])
            ],
            # go.Scatter(x=r_grs[k], y=gs[k, 0, 1]),
            traces=[0, 1, 2] + list(range(3, 3 + gs.shape[1] * gs.shape[2])),
            name=k,
        )
        for k in range(len(traj))
    ]

    fig.frames = frames

    play_buttons = [
        dict(
            label="Play",
            method="animate",
            args=[
                None,
                dict(
                    frame=dict(duration=30, redraw=True),
                    transition=dict(duration=0),
                    fromcurrent=True,
                    mode="immediate",
                ),
            ],
        ),
        dict(
            label="Pause",
            method="animate",
            args=[
                [None],
                dict(
                    frame=dict(duration=0, redraw=True),
                    transition=dict(duration=0),
                    fromcurrent=True,
                    mode="immediate",
                ),
            ],
        ),
    ]

    # Create and add slider

    steps = []
    for f in frames:
        step = dict(
            method="animate",
            label=f.name,
            args=[
                [f.name],
                {
                    "frame": {"duration": 30, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 0},
                },
            ],
        )
        steps.append(step)

    sliders = [
        dict(
            active=0,
            currentvalue={"prefix": "Frame: "},
            pad={"t": 50},
            transition={"duration": 0},
            steps=steps,
        )
    ]

    trace_buttons = (
        [
            {
                "label": f"{s1}-{s2}",
                "method": "restyle",
                "args": [{"visible": "legendonly"}, [3 + s1 * gs.shape[1] + s2]],
                "args2": [{"visible": True}, [3 + s1 * gs.shape[1] + s2]],
            }
            for s1 in range(gs.shape[1])
            for s2 in range(gs.shape[2])
        ]
        + [
            {
                "label": "All",
                "method": "restyle",
                "args": [
                    {"visible": [True] * 3 + [True] * (gs.shape[1] * gs.shape[2])}
                ],
            }
        ]
        + [
            {
                "label": "None",
                "method": "restyle",
                "args": [
                    {
                        "visible": [True] * 3
                        + ["legendonly"] * (gs.shape[1] * gs.shape[2])
                    }
                ],
            }
        ]
    )

    fig.update_layout(
        title="MD Trajectory and Energy",
        height=800,
        width=1000,
        xaxis_title="x",
        yaxis_title="y",
        font=dict(family="Arial", size=14),
        paper_bgcolor="rgba(255,255,255,0.9)",
        plot_bgcolor="rgba(0,0,0,0)",
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                showactive=False,
                y=0.7,
                x=1.05,
                xanchor="left",
                yanchor="top",
                buttons=play_buttons,
            ),
            dict(
                type="buttons",
                direction="left",
                showactive=False,
                y=0.5,
                x=1.05,
                xanchor="left",
                yanchor="top",
                buttons=trace_buttons,
            ),
        ],
        sliders=sliders,
    )

    # fig.update_layout(
    #     updatemenus=[
    #         dict(
    #             type="buttons",
    #             direction="left",
    #             showactive=False,
    #             y=0.5,
    #             x=1.05,
    #             xanchor="left",
    #             yanchor="top",
    #             buttons=trace_buttons,
    #         )
    #     ],
    # )

    # Make the trajectory plot have a square aspect ratio
    fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_xaxes(
        constrain="domain",  # compresses the xaxis by decreasing its domain
    )
    fig.update_yaxes(range=[0, np.max(gs)], row=2, col=1)

    del es, r_grs, gs

    return fig


def plot_rdf(displacement_fn, r, r_min=0.1, r_max=2.0, n_bins=1000, dr=0.01):
    eval_radii, pair_corr = calc_rdf(
        displacement_fn, r, r_min=0.1, r_max=2.0, n_bins=n_bins, dr=dr, total_rdf=True
    )

    plt.plot(eval_radii, pair_corr, "k", linewidth=3)

    plt.xlabel("r", fontsize=20)
    plt.ylabel("g(r)", fontsize=20)

    plt.legend(["Raft", "Distorted Raft"], loc="upper right")
    plt.xlim([np.min(eval_radii), np.max(eval_radii)])

    plt.show()


def plot_loss_grad_series(
    xs, losses, grads, losses_std=None, grads_std=None, labels=None
):
    fig, axs = plt.subplots(2, 1, figsize=(6, 8), dpi=300)
    fig.subplots_adjust(hspace=0.25)

    if labels is None:
        labels = range(len(losses))

    assert len(labels) == len(losses)

    if losses_std is not None and grads_std is not None:
        for i in range(len(losses)):
            axs[0].plot(xs, losses[i], linewidth=2, label=labels[i])
            axs[0].fill_between(
                xs,
                losses[i] + losses_std[i],
                losses[i] - losses_std[i],
                alpha=0.4,
            )
            axs[1].plot(xs, grads[i], linewidth=2, label=labels[i])
            axs[1].fill_between(
                xs,
                grads[i] + grads_std[i],
                grads[i] - grads_std[i],
                alpha=0.4,
            )

    else:
        for i in range(len(losses)):
            axs[0].plot(xs, losses[i], linewidth=2, label=labels[i])
            axs[1].plot(xs, grads[i], linewidth=2, label=labels[i])

    # xmin, xmax = 0.9, 1.1
    axs[0].set_ylabel("Loss")
    axs[0].set_xlabel("$r_e$")
    # axs[0].set_xlim([xmin, xmax])
    # axs[0].set_xlim([xmin, xmax])
    # axs[0].set_yscale('log')
    axs[1].axhline(y=0.0, color="k", linestyle="--")
    axs[1].set_ylabel("grad(Loss)")
    axs[1].set_xlabel("$r_e$")
    # axs[1].set_xlim([xmin, xmax])
    # axs[1].set_yscale('log')
    # plt.legend(title="Grad. \nsimulation \nsteps", bbox_to_anchor=(1.05, 1.4))
    handles, labels = axs[0].get_legend_handles_labels()
    legend_ax = fig.add_subplot(111, frameon=False)
    legend_ax.tick_params(
        labelcolor="none",
        which="both",
        top=False,
        bottom=False,
        left=False,
        right=False,
    )

    legend_ax.legend(
        handles,
        labels,
        title="In-gradient \nsimulation \nsteps",
        loc="center left",
        fontsize=8,
        title_fontsize=8,
        bbox_to_anchor=(1.04, 0.5),
    )
    return fig, axs


def plot_phase_diagram(img_fp, temp_units="C", **subplot_kwargs):
    img = mpimg.imread(img_fp)

    fig, ax = plt.subplots(**subplot_kwargs)

    x_start = 211
    x_stop = 1751
    scale_x = (x_stop - x_start) / 100
    y_start = 49
    y_stop = 1198
    scale_y = (y_stop - y_start) / 100

    # Plot phase diagram
    if temp_units == "C":
        imgplot = ax.imshow(
            img[y_start:y_stop, x_start:x_stop],
            extent=[0, 100, 100, 1100],
            aspect="auto",
        )
    else:
        imgplot = ax.imshow(
            img[y_start:y_stop, x_start:x_stop],
            extent=[0, 100, 100 + 273, 1100 + 273],
            aspect="auto",
        )
    # ax.set_xticks(np.linspace(0,x_stop-x_start, 11), [f"{i:2.0f}" for i in np.arange(0,110,10)])
    ax.set_xlabel("Composition, $c$ / at. % Cu")

    # ax.set_ylim(1100,300)
    # ax.invert_yaxis()
    # ax.set_yticks(np.linspace(y_stop-y_start, 0, 11), [f"{i:2.0f}" for i in
    # np.arange(100,1200,100)])
    if temp_units == "C":
        ax.set_ylabel("$T$ / $^{\\circ}\\text{C}$")
    else:
        ax.set_ylabel("$T$/$\\text{K}$")
    # ax.set_title("Cu-Au Binary Phase Diagram")

    return fig, ax, scale_x, scale_y


# DEPRECATED
# def draw_system(R, box_size, marker_size, color=None):
#     if color is None:
#         color = [64 / 256] * 3
#     color = onp.array(color)
#     ms = marker_size / box_size

#     R = onp.array(R)

#     marker_style = dict(
#         linestyle="none",
#         markeredgewidth=3,
#         marker="o",
#         markersize=ms,
#         color=color,
#         fillstyle="none",
#     )

#     plt.plot(R[:, 0], R[:, 1], **marker_style)
#     plt.plot(R[:, 0] + box_size, R[:, 1], **marker_style)
#     plt.plot(R[:, 0], R[:, 1] + box_size, **marker_style)
#     plt.plot(R[:, 0] + box_size, R[:, 1] + box_size, **marker_style)
#     plt.plot(R[:, 0] - box_size, R[:, 1], **marker_style)
#     plt.plot(R[:, 0], R[:, 1] - box_size, **marker_style)
#     plt.plot(R[:, 0] - box_size, R[:, 1] - box_size, **marker_style)

#     plt.xlim([0, box_size])
#     plt.ylim([0, box_size])
#     plt.axis("off")
