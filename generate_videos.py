#! python3

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import math
import numpy as np
import matplotlib.animation as animation

VICON_FPS = 100
VIDEO_FPS = 20

start = 0 * VIDEO_FPS
end = 10 * VIDEO_FPS  # 10sec video

#####################################################

csv_files = Path(".").glob("study6H.csv")
data_df = pd.concat([pd.read_csv(fp) for fp in csv_files])


# convert df index to time periods
data_df.set_index(
    pd.date_range(0, periods=len(data_df), freq="%fS" % (1 / VICON_FPS)), inplace=True
)

# resample df to target FPS
data_df = data_df.resample("%fS" % (1 / VIDEO_FPS)).mean()

df_dist = pd.DataFrame()
OneTX, OneTY, OneTZ = data_df["1TX"], data_df["1TY"], data_df["1TZ"]
TwoTX, TwoTY, TwoTZ = data_df["2TX"], data_df["2TY"], data_df["2TZ"]
ThreeTX, ThreeTY, ThreeTZ = data_df["3TX"], data_df["3TY"], data_df["3TZ"]
FourTX, FourTY, FourTZ = data_df["4TX"], data_df["4TY"], data_df["4TZ"]
FiveTX, FiveTY, FiveTZ = data_df["5TX"], data_df["5TY"], data_df["5TZ"]

df_dist["Frame"] = data_df["Frame"]
df_dist["distance_1_2"] = np.power(
    np.power(OneTX - TwoTX, 2) + np.power(OneTY - TwoTY, 2), 0.5
)
df_dist["distance_1_3"] = np.power(
    np.power(OneTX - ThreeTX, 2) + np.power(OneTY - ThreeTY, 2), 0.5
)
df_dist["distance_1_4"] = np.power(
    np.power(OneTX - FourTX, 2) + np.power(OneTY - FourTY, 2), 0.5
)
df_dist["distance_1_5"] = np.power(
    np.power(OneTX - FiveTX, 2) + np.power(OneTY - FiveTY, 2), 0.5
)

df_dist["distance_2_1"] = np.power(
    np.power(TwoTX - OneTX, 2) + np.power(TwoTY - OneTY, 2), 0.5
)
df_dist["distance_2_3"] = np.power(
    np.power(TwoTX - ThreeTX, 2) + np.power(TwoTY - ThreeTY, 2), 0.5
)
df_dist["distance_2_4"] = np.power(
    np.power(TwoTX - FourTX, 2) + np.power(TwoTY - FourTY, 2), 0.5
)
df_dist["distance_2_5"] = np.power(
    np.power(TwoTX - FiveTX, 2) + np.power(TwoTY - FiveTY, 2), 0.5
)

df_dist["distance_3_1"] = np.power(
    np.power(ThreeTX - OneTX, 2) + np.power(ThreeTY - OneTY, 2), 0.5
)
df_dist["distance_3_2"] = np.power(
    np.power(ThreeTX - TwoTX, 2) + np.power(ThreeTY - TwoTY, 2), 0.5
)
df_dist["distance_3_4"] = np.power(
    np.power(ThreeTX - FourTX, 2) + np.power(ThreeTY - FourTY, 2), 0.5
)
df_dist["distance_3_5"] = np.power(
    np.power(ThreeTX - FiveTX, 2) + np.power(ThreeTY - FiveTY, 2), 0.5
)

df_dist["distance_4_1"] = np.power(
    np.power(FourTX - OneTX, 2) + np.power(FourTY - OneTY, 2), 0.5
)
df_dist["distance_4_2"] = np.power(
    np.power(FourTX - TwoTX, 2) + np.power(FourTY - TwoTY, 2), 0.5
)
df_dist["distance_4_3"] = np.power(
    np.power(FourTX - ThreeTX, 2) + np.power(FourTY - ThreeTY, 2), 0.5
)
df_dist["distance_4_5"] = np.power(
    np.power(FourTX - FiveTX, 2) + np.power(FourTY - FiveTY, 2), 0.5
)

df_dist["distance_5_1"] = np.power(
    np.power(FiveTX - OneTX, 2) + np.power(FiveTY - OneTY, 2), 0.5
)
df_dist["distance_5_2"] = np.power(
    np.power(FiveTX - TwoTX, 2) + np.power(FiveTY - TwoTY, 2), 0.5
)
df_dist["distance_5_3"] = np.power(
    np.power(FiveTX - ThreeTX, 2) + np.power(FiveTY - ThreeTY, 2), 0.5
)
df_dist["distance_5_4"] = np.power(
    np.power(FiveTX - FourTX, 2) + np.power(FiveTY - FourTY, 2), 0.5
)


fov_angle = math.radians(60)

length = 500

fig, ax = plt.subplots(figsize=(10, 10))


def init():
    plt.legend(loc="upper left")

    plt.xlim(-5000, 5000)
    plt.ylim(-5000, 5000)


def update(frame):

    print("Done %s%%" % (100 * (frame - start) / (end - start)))

    row = data_df.iloc[frame]

    x = row["1TX"]
    y = row["1TY"]
    yaw = row["1RZ"]

    fig.clear()

    plt.xlim(-5000, 5000)
    plt.ylim(-5000, 5000)

    plt.scatter(x, y, s=100, label="1")
    plt.arrow(
        x,
        y,
        length * np.cos(yaw),
        length * np.sin(yaw),
        head_width=100,
        head_length=100,
        color="#82ad9d",
        lw=1,
    )
    plt.arrow(
        x,
        y,
        length * np.cos(yaw - fov_angle),
        length * np.sin(yaw - fov_angle),
        head_width=100,
        head_length=100,
        color="#82ad9d",
        lw=1,
    )
    plt.arrow(
        x,
        y,
        length * np.cos(yaw + fov_angle),
        length * np.sin(yaw + fov_angle),
        head_width=100,
        head_length=100,
        color="#82ad9d",
        lw=1,
    )

    x = row["2TX"]
    y = row["2TY"]
    yaw = row["2RZ"]
    plt.scatter(x, y, s=100, label="2")
    plt.arrow(
        x,
        y,
        length * np.cos(yaw),
        length * np.sin(yaw),
        head_width=100,
        head_length=100,
        color="#82ad9d",
        lw=1,
    )
    plt.arrow(
        x,
        y,
        length * np.cos(yaw - fov_angle),
        length * np.sin(yaw - fov_angle),
        head_width=100,
        head_length=100,
        color="#82ad9d",
        lw=1,
    )
    plt.arrow(
        x,
        y,
        length * np.cos(yaw + fov_angle),
        length * np.sin(yaw + fov_angle),
        head_width=100,
        head_length=100,
        color="#82ad9d",
        lw=1,
    )

    x = row["3TX"]
    y = row["3TY"]
    yaw = row["3RZ"]
    plt.scatter(x, y, s=100, label="3")
    plt.arrow(
        x,
        y,
        length * np.cos(yaw),
        length * np.sin(yaw),
        head_width=100,
        head_length=100,
        color="#82ad9d",
        lw=1,
    )
    plt.arrow(
        x,
        y,
        length * np.cos(yaw - fov_angle),
        length * np.sin(yaw - fov_angle),
        head_width=100,
        head_length=100,
        color="#82ad9d",
        lw=1,
    )
    plt.arrow(
        x,
        y,
        length * np.cos(yaw + fov_angle),
        length * np.sin(yaw + fov_angle),
        head_width=100,
        head_length=100,
        color="#82ad9d",
        lw=1,
    )

    x = row["4TX"]
    y = row["4TY"]
    yaw = row["4RZ"]
    plt.scatter(x, y, s=100, label="4")
    plt.arrow(
        x,
        y,
        length * np.cos(yaw),
        length * np.sin(yaw),
        head_width=100,
        head_length=100,
        color="#82ad9d",
        lw=1,
    )
    plt.arrow(
        x,
        y,
        length * np.cos(yaw - fov_angle),
        length * np.sin(yaw - fov_angle),
        head_width=100,
        head_length=100,
        color="#82ad9d",
        lw=1,
    )
    plt.arrow(
        x,
        y,
        length * np.cos(yaw + fov_angle),
        length * np.sin(yaw + fov_angle),
        head_width=100,
        head_length=100,
        color="#82ad9d",
        lw=1,
    )

    x = row["5TX"]
    y = row["5TY"]
    yaw = row["5RZ"]
    plt.scatter(x, y, s=100, label="5")
    plt.arrow(
        x,
        y,
        length * np.cos(yaw),
        length * np.sin(yaw),
        head_width=100,
        head_length=100,
        color="#82ad9d",
        lw=1,
    )
    plt.arrow(
        x,
        y,
        600 * np.cos(yaw - fov_angle),
        600 * np.sin(yaw - fov_angle),
        head_width=100,
        head_length=100,
        color="#82ad9d",
        lw=1,
    )
    plt.arrow(
        x,
        y,
        600 * np.cos(yaw + fov_angle),
        600 * np.sin(yaw + fov_angle),
        head_width=100,
        head_length=100,
        color="#82ad9d",
        lw=1,
    )


ani = animation.FuncAnimation(fig, update, frames=range(start, end), init_func=init)
# plt.show()
ani.save("test.mp4", fps=VIDEO_FPS)
