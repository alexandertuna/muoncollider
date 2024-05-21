import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots

pqname = "writeCaloHits.parquet"
df_all = pd.read_parquet(pqname)

# https://github.com/MuonColliderSoft/lcgeo/blob/master/MuColl/MuColl_v1.1/config.xml
HCalEndcap_outer_radius = 3246.0
HCalEndcap_min_z = 2539.0
HCalEndcap_max_z = 4129.0
n_points = 100
n_dim = 3

radius = HCalEndcap_outer_radius
phis = np.linspace(0, 2 * np.pi, n_points)
x_endcap_a = radius * np.cos(phis)
y_endcap_a = radius * np.sin(phis)
z_endcap_a = np.ones(n_points)*HCalEndcap_max_z
x_endcap_c = radius * np.cos(phis)
y_endcap_c = radius * np.sin(phis)
z_endcap_c = -np.ones(n_points)*HCalEndcap_max_z

x_barrel_mid = np.zeros(n_points)
y_barrel_mid = np.zeros(n_points)
z_barrel_mid = np.linspace(-HCalEndcap_max_z, HCalEndcap_max_z, n_points)
x_barrel_top = np.ones(n_points)*HCalEndcap_outer_radius
y_barrel_top = np.zeros(n_points)
z_barrel_top = np.linspace(-HCalEndcap_max_z, HCalEndcap_max_z, n_points)
x_barrel_bot = -np.ones(n_points)*HCalEndcap_outer_radius
y_barrel_bot = np.zeros(n_points)
z_barrel_bot = np.linspace(-HCalEndcap_max_z, HCalEndcap_max_z, n_points)

rows = 3
cols = 1
specs = [
    [{"type": "scatter3d"}] * cols
] * rows
fig = make_subplots(rows=rows, cols=cols, specs=specs)

for row in range(rows):

    condition = df_all["event"] == row
    df = df_all[condition]

    cond = df["hit_system"] >= 20
    df_ecal = df[cond]
    df_hcal = df[~cond]

    ecal = dict(size=2, color="#ff0000")
    hcal = dict(size=2, color="#0000ff")
    mode = "lines"
    outline = dict(width=2, color="#000000")
    for col in range(cols):
        fig.add_trace(go.Scatter3d(x=df_hcal.hit_x,
                                   y=df_hcal.hit_y,
                                   z=df_hcal.hit_z,
                                   mode="markers",
                                   marker=hcal,
                               ), row=row+1, col=col+1)
        fig.add_trace(go.Scatter3d(x=df_ecal.hit_x,
                                   y=df_ecal.hit_y,
                                   z=df_ecal.hit_z,
                                   mode="markers",
                                   marker=ecal,
                               ), row=row+1, col=col+1)
        if col == 0:
            fig.add_trace(go.Scatter3d(x=x_barrel_mid, y=y_barrel_mid, z=z_barrel_mid, mode=mode, line=outline), row=row+1, col=col+1)
            fig.add_trace(go.Scatter3d(x=x_endcap_a, y=y_endcap_a, z=z_endcap_a, mode=mode, line=outline), row=row+1, col=col+1)
            fig.add_trace(go.Scatter3d(x=x_endcap_c, y=y_endcap_c, z=z_endcap_c, mode=mode, line=outline), row=row+1, col=col+1)


# fig.show()

camera = dict(
    up=dict(x=1, y=0, z=0),
    eye=dict(x=1.25, y=1.25, z=-1.25)
)

fig.update_layout(height=700*rows, showlegend=False)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.update_layout(scene_camera=camera)
for key in fig.layout:
    if key.startswith("scene"):
        fig.layout[key].camera = camera
fig.write_html("neutrons.html")
