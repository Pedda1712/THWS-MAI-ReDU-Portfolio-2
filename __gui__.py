"""
A GUI to control the parameters of the
bouncing ball particle filter.
"""
from tkinter import *

from Simulation import SimulationParameters, Simulation

def par_field(parent, x, y, text, ival="-"):
    label = Label(parent, text=text, relief="sunken")
    myvar = StringVar()
    entry = Entry(parent, width=5,textvariable=myvar)
    entry.insert(END, ival)
    label.grid(row=y, column=x*2, sticky="ew")
    entry.grid(row=y, column=x*2+1, sticky="ew")
    return myvar

def par_check(parent, x, y, text, ival=True):
    label = Label(parent, text=text, relief="sunken")
    CheckVar1 = IntVar()
    entry = Checkbutton(parent, text="", variable = CheckVar1)
    if ival:
        entry.select()
    label.grid(row=y, column=x*2, sticky="ew")
    entry.grid(row=y, column=x*2+1, sticky="ew")
    return CheckVar1

def run_app(p = SimulationParameters()):
    root = Tk()
    root.title("Ball Arena Control Panel")

    pframe = Frame(root)

    actual_container = LabelFrame(pframe, text="Actual World Parmeters")
    a_width =   par_field(actual_container, 0, 0, "Width", p.width)
    a_height =  par_field(actual_container, 1, 0, "Height", p.height)
    a_radius =  par_field(actual_container, 0, 1, "Ball Radius", p.ball_radius)
    a_numb   =  par_field(actual_container, 1, 1, "Ball Count", p.number_of_balls)
    a_gravity = par_field(actual_container, 0, 2, "Gravity", p.gravity)
    a_bounce  = par_field(actual_container, 1, 2, "Bounce Discount", p.bounce_discount)
    a_air     = par_field(actual_container, 0, 3, "Air Discount", p.air_discount)
    a_ground  = par_field(actual_container, 1, 3, "Ground Discount", p.ground_discount)
    a_svarx   = par_field(actual_container, 0, 4, "Sensor Variance X", p.sensor_variance[0])
    a_svary   = par_field(actual_container, 1, 4, "Sensor Variance Y", p.sensor_variance[1])
    a_ivarx   = par_field(actual_container, 0, 5, "Initial Velocity Variance X", p.initial_velocity_variance[0])
    a_ivary   = par_field(actual_container, 1, 5, "Initial Velocity Variance Y", p.initial_velocity_variance[1])
    
    actual_container.pack(side = "left", anchor="n")

    assumed_container = LabelFrame(pframe, text="Assumed World Parameters")
    s_width =   par_field(assumed_container, 0, 0, "Width", p.assumed_width)
    s_height =  par_field(assumed_container, 1, 0, "Height", p.assumed_height)
    s_radius =  par_field(assumed_container, 0, 1, "Ball Radius", p.assumed_ball_radius)
    s_numb   =  par_field(assumed_container, 1, 1, "Ball Count", p.assumed_number_of_balls)
    s_gravity = par_field(assumed_container, 0, 2, "Gravity", p.assumed_gravity)
    s_bounce  = par_field(assumed_container, 1, 2, "Bounce Discount", p.assumed_bounce_discount)
    s_air     = par_field(assumed_container, 0, 3, "Air Discount", p.assumed_air_discount)
    s_ground  = par_field(assumed_container, 1, 3, "Ground Discount", p.assumed_ground_discount)
    s_svarx   = par_field(assumed_container, 0, 4, "Sensor Variance X", p.assumed_sensor_variance[0])
    s_svary   = par_field(assumed_container, 1, 4, "Sensor Variance Y", p.assumed_sensor_variance[1])
    s_ivarx   = par_field(assumed_container, 0, 5, "Initial Velocity Variance X", p.assumed_initial_velocity_variance[0])
    s_ivary   = par_field(assumed_container, 1, 5, "Initial Velocity Variance Y", p.assumed_initial_velocity_variance[1])
    s_tvarx   = par_field(assumed_container, 0, 6, "Transition Velocity Variance X", p.transition_velocity_variance[0])
    s_tvary   = par_field(assumed_container, 1, 6, "Transition Velocity Variance Y", p.transition_velocity_variance[1])
    assumed_container.pack(side = "left", anchor="n")

    meta_lframe = LabelFrame(root, text="Simulation Parameters")
    meta_container = Frame(meta_lframe)
    m_mps     = par_field(meta_container, 0, 0, "Measurements Per Second", p.measurements_per_second)
    m_pnum    = par_field(meta_container, 1, 0, "Particle Count", p.number_of_particles)
    m_seed    = par_field(meta_container, 0, 1, "Seed", p.seed)
    m_maxs    = par_field(meta_container, 1, 1, "Steps", p.max_steps)
    m_ls      = par_check(meta_container, 0, 2, "Show PyGame Visualisation", p.live_show)
    m_sp      = par_check(meta_container, 0, 3, "Show Particles", p.show_particles)
    m_so      = par_check(meta_container, 1, 3, "Show Observations", p.show_observations)
    m_sa      = par_check(meta_container, 1, 2, "Show Actual Positions", p.show_actual_positions)
    m_ss      = par_check(meta_container, 0, 5, "Show Summary Plots", p.show_summary_plots)
    m_tails   = par_field(meta_container, 1, 5, "Tail Length", p.visualize_tail_length)

    def run():
        root.destroy()
        p = SimulationParameters(
            int(a_numb.get()),
            int(a_width.get()),
            int(a_height.get()),
            float(a_gravity.get()),
            float(a_radius.get()),
            float(a_bounce.get()),
            float(a_air.get()),
            float(a_ground.get()),
            (float(a_svarx.get()), float(a_svary.get())),
            (float(a_ivarx.get()), float(a_ivary.get())),
            int(s_numb.get()),
            int(s_width.get()),
            int(s_height.get()),
            float(s_gravity.get()),
            float(s_radius.get()),
            float(s_bounce.get()),
            float(s_air.get()),
            float(s_ground.get()),
            (float(s_svarx.get()), float(s_svary.get())),
            (float(s_ivarx.get()), float(s_ivary.get())),
            (float(s_tvarx.get()), float(s_tvary.get())),
            int(m_mps.get()),
            int(m_pnum.get()),
            int(m_seed.get()),
            bool(m_ls.get()),
            int(m_tails.get()),
            int(m_maxs.get()),
            bool(m_sp.get()),
            bool(m_so.get()),
            bool(m_sa.get()),
            bool(m_ss.get())
        )
        Simulation(p).run()
        run_app(p) # this leaks (probably), doesn't matter right now
    
    go = Button(meta_lframe, text="GO", bg="red", command = run)
    meta_container.pack(anchor="w", side="left")
    go.pack(expand=True, fill="both", side="left")

    pframe.pack()
    meta_lframe.pack(expand=True, fill="both")

    root.resizable(False, False)
    root.mainloop()


if __name__ == "__main__":
    run_app()
